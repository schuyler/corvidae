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
    assert len(warnings) == 1 and str(path) in warnings[0] and "check the directory" in warnings[0]

    # Once the directory works again, the next write rotates and logging recovers.
    monkeypatch.undo()
    appender.write_line("record-06")
    appender.close()
    assert _read(tmp_path / "chan.jsonl.1")[0] == "record-00" and _read(path) == ["record-06"]


def test_a_failed_rename_leaves_the_backup_chain_intact(tmp_path, caplog, monkeypatch):
    """The live file is the one that cannot be renamed (#606): the chain of
    backups is left as it was, rather than shifted one step per retry."""
    import corvidae.jsonl_rotate as mod

    path = tmp_path / "chan.jsonl"
    before = {}
    for n in range(1, 5):
        backup = path.with_name(f"{path.name}.{n}")
        backup.write_text(f"backup-{n}\n")
        before[backup] = backup.read_text()
    path.write_text("x" * 25 + "\n")

    appender = RotatingAppender(path, max_bytes=30, backup_count=4)
    real_replace = mod.os.replace

    def deny_the_live_file(src, dst, *args, **kwargs):
        # EPERM on the live file's rename, as `chattr +a` gives; the chain's own
        # renames still work.
        if str(src) == str(path):
            raise PermissionError(1, "Operation not permitted", str(src))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(mod.os, "replace", deny_the_live_file)
    with caplog.at_level(logging.WARNING, logger="corvidae.jsonl_rotate"):
        for i in range(3):
            appender.write_line(f"record-{i}")

    monkeypatch.undo()
    for backup, contents in before.items():
        assert backup.read_text() == contents, f"{backup.name} was shifted by a failed rotation"


def test_a_second_failure_streak_warns_again(tmp_path, caplog, monkeypatch):
    """A streak that ends without a successful rotation -- the file falls back
    under the cap -- does not silence the next one (#606)."""
    import corvidae.jsonl_rotate as mod

    path = tmp_path / "chan.jsonl"
    appender = RotatingAppender(path, max_bytes=20, backup_count=2)

    def denied(*args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(mod.os, "replace", denied)

    def warnings() -> list[str]:
        return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]

    with caplog.at_level(logging.WARNING, logger="corvidae.jsonl_rotate"):
        for i in range(4):
            appender.write_line(f"record-{i}")
        first_streak = len(warnings())

        # Under the cap again: an external truncate, which the size check sees.
        path.write_text("")
        appender.write_line("after-the-truncate")
        for i in range(4):
            appender.write_line(f"again-{i}")
        second_streak = len(warnings())

    monkeypatch.undo()
    appender.close()
    assert first_streak == 1, "one warning per streak"
    assert second_streak == 2, "the second streak is reported, not silent"


def test_a_failure_part_way_through_the_chain_puts_every_backup_back(tmp_path, caplog, monkeypatch):
    """The failure need not be the first rename (#606): whatever the attempt had
    already shifted is replayed backwards, so `.1` through `.4` still hold the
    data they held before, and the live file keeps its own records."""
    import corvidae.jsonl_rotate as mod

    path = tmp_path / "chan.jsonl"
    before = {}
    for n in range(1, 5):
        backup = path.with_name(f"{path.name}.{n}")
        backup.write_text(f"backup-{n}\n")
        before[backup] = backup.read_text()
    path.write_text("x" * 25 + "\n")

    appender = RotatingAppender(path, max_bytes=30, backup_count=4)
    real_replace = mod.os.replace

    def deny_the_first_backup(src, dst, *args, **kwargs):
        # `.1` lands last, after `.3`->`.4` and `.2`->`.3` have already happened.
        if str(dst).endswith(".1") and str(src).endswith(".rotating-" + str(mod.os.getpid())):
            raise PermissionError(1, "Operation not permitted", str(dst))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(mod.os, "replace", deny_the_first_backup)
    with caplog.at_level(logging.WARNING, logger="corvidae.jsonl_rotate"):
        for i in range(3):
            appender.write_line(f"record-{i}")
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]

    monkeypatch.undo()
    for backup, contents in before.items():
        assert backup.read_text() == contents, f"{backup.name} was not put back"
    live = _read(path)
    assert live[0] == "x" * 25 and live[1:] == [f"record-{i}" for i in range(3)]
    assert len(warnings) == 1 and str(path) in warnings[0]
    # No temporary file outlives the attempt.
    assert [p.name for p in tmp_path.iterdir() if "rotating" in p.name or "dropping" in p.name] == []


def test_a_partial_rotation_heals_on_the_next_successful_one(tmp_path, monkeypatch):
    """After a failed attempt the chain is where it was, so the next write that
    can rotate produces the same chain a first-time rotation would."""
    import corvidae.jsonl_rotate as mod

    path = tmp_path / "chan.jsonl"
    for n in range(1, 3):
        path.with_name(f"{path.name}.{n}").write_text(f"backup-{n}\n")
    path.write_text("y" * 25 + "\n")

    appender = RotatingAppender(path, max_bytes=30, backup_count=2)
    real_replace = mod.os.replace
    monkeypatch.setattr(mod.os, "replace", lambda *a, **k: (_ for _ in ()).throw(PermissionError(13, "denied")))
    appender.write_line("record-0")
    monkeypatch.setattr(mod.os, "replace", real_replace)

    appender.write_line("record-1")
    appender.close()
    assert _read(path) == ["record-1"]
    assert _read(path.with_name("chan.jsonl.1"))[0] == "y" * 25
    assert _read(path.with_name("chan.jsonl.2")) == ["backup-1"]
