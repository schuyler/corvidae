"""The JSONL sinks are capped in-process (#404).

The transcripts under ``daemon.jsonl_log_dir`` and ``daemon.metrics_jsonl``
grew without bound; they now rotate like ``logging.file`` does.
"""

import json
import logging

import pytest

from corvidae.jsonl_rotate import (
    DEFAULT_BACKUP_COUNT,
    DEFAULT_MAX_BYTES,
    RotatingAppender,
    parse_rotation,
)


def _read(path) -> list[str]:
    return path.read_text().splitlines()


def test_rotation_keeps_the_newest_backups_in_order_and_logs_the_file(tmp_path, caplog):
    path = tmp_path / "chan.jsonl"
    appender = RotatingAppender(path, max_bytes=30, backup_count=2)
    with caplog.at_level(logging.INFO, logger="corvidae.jsonl_rotate"):
        for i in range(10):
            appender.write_line(f"record-{i:02d}")  # 10 bytes with its newline
    appender.close()

    # Three records fit per file: the live file and two backups, no third.
    assert [_read(f) for f in (path, tmp_path / "chan.jsonl.1", tmp_path / "chan.jsonl.2")] == [
        ["record-09"], ["record-06", "record-07", "record-08"], ["record-03", "record-04", "record-05"],
    ]
    assert not (tmp_path / "chan.jsonl.3").exists()
    assert any("chan.jsonl" in r.getMessage() and r.levelno == logging.INFO for r in caplog.records)


def test_a_file_already_over_the_cap_rotates_on_its_first_write(tmp_path):
    """A deployment's existing 14 MB transcript is rotated, not appended to."""
    path = tmp_path / "chan.jsonl"
    path.write_text("old-" * 20 + "\n")
    appender = RotatingAppender(path, max_bytes=50, backup_count=1)
    appender.write_line("new")
    appender.close()

    assert _read(path) == ["new"]
    assert _read(tmp_path / "chan.jsonl.1") == ["old-" * 20]


@pytest.mark.parametrize("max_bytes, backups", [(0, 5), (10, 0)])
def test_zero_disables_rotation_or_keeps_no_backups(tmp_path, max_bytes, backups):
    path = tmp_path / "chan.jsonl"
    appender = RotatingAppender(path, max_bytes=max_bytes, backup_count=backups)
    for i in range(5):
        appender.write_line(f"record-{i:02d}")
    appender.close()

    assert not (tmp_path / "chan.jsonl.1").exists()
    assert _read(path)[-1] == "record-04"
    assert (len(_read(path)) == 5) == (max_bytes == 0)  # unbounded vs truncated


def test_defaults_match_logging_file():
    assert parse_rotation({}) == (DEFAULT_MAX_BYTES, DEFAULT_BACKUP_COUNT) == (10_485_760, 5)
    cfg = {"logging": {"jsonl": {"max_bytes": 100, "backup_count": 1}}}
    assert parse_rotation(cfg) == (100, 1)


@pytest.mark.parametrize("bad", [{"max_bytes": -1}, {"backup_count": "five"}, {"max_bytes": True}])
def test_a_bad_value_names_the_key_the_value_and_what_is_accepted(bad):
    (key, value), = bad.items()
    with pytest.raises(ValueError) as exc:
        parse_rotation({"logging": {"jsonl": bad}})

    message = str(exc.value)
    assert f"logging.jsonl.{key}" in message and repr(value) in message
    assert "integer" in message and "0 or greater" in message


# The two plugins read the same keys and rotate their own file.
async def test_both_sinks_rotate(tmp_path):
    from corvidae.context import MessageType
    from corvidae.jsonl_log import JsonlLogPlugin
    from corvidae.metrics import MetricsJsonlPlugin

    config = {
        "daemon": {"jsonl_log_dir": str(tmp_path / "logs"), "metrics_jsonl": str(tmp_path / "m.jsonl")},
        "logging": {"jsonl": {"max_bytes": 200, "backup_count": 1}},
    }
    transcript, metrics = JsonlLogPlugin(None), MetricsJsonlPlugin(None)
    for plugin in (transcript, metrics):
        await plugin.on_init(pm=None, config=config)
        await plugin.on_start(config=config)
    channel = type("Ch", (), {"id": "cli:default"})()
    for i in range(10):
        await transcript.persist_message(channel, {"role": "user", "content": f"hello {i}"}, MessageType.MESSAGE)
        await metrics.on_metrics(name="llm.latency_ms", value=float(i), tags={})
    await transcript.on_stop()
    await metrics.on_stop()

    for live in (tmp_path / "logs" / "cli_default.jsonl", tmp_path / "m.jsonl"):
        assert live.stat().st_size <= 200 and live.with_name(live.name + ".1").exists()
        assert not live.with_name(live.name + ".2").exists()
        assert all(json.loads(line) for line in _read(live))


def test_a_failing_rotation_keeps_appending_and_warns_once(tmp_path, caplog, monkeypatch):
    """The directory is unwritable, so rotation cannot rename: the file grows past
    the cap instead of dropping every later record, as it did before the cap."""
    import corvidae.jsonl_rotate as mod

    path = tmp_path / "chan.jsonl"
    appender = RotatingAppender(path, max_bytes=30, backup_count=2)

    def denied(*args, **kwargs):
        raise PermissionError("read-only directory")

    monkeypatch.setattr(mod.os, "replace", denied)
    with caplog.at_level(logging.WARNING, logger="corvidae.jsonl_rotate"):
        for i in range(6):
            appender.write_line(f"record-{i:02d}")
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]

    assert len(_read(path)) == 6, "no record is lost to the failed rotation"
    assert len(warnings) == 1 and str(path) in warnings[0] and "permission" in warnings[0].lower()

    # Once the directory works again, the next write rotates and logging recovers.
    monkeypatch.undo()
    appender.write_line("record-06")
    appender.close()
    assert _read(tmp_path / "chan.jsonl.1")[0] == "record-00" and _read(path) == ["record-06"]
