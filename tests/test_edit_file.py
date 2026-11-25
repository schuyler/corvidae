"""edit_file: an exact-passage edit that refuses when it cannot be sure (#1160)."""

from __future__ import annotations

import asyncio

import pytest

from corvidae.tool_policy import expand_disabled
from corvidae.tools.files import edit_file


async def test_an_exact_passage_is_replaced_and_the_rest_of_the_file_kept(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# Title\n\nfirst line\nsecond line\n")

    result = await edit_file(str(path), "first line", "first line, edited")

    assert result.startswith(f"Edited {path}: replaced 1 occurrence at line 3")
    assert path.read_text() == "# Title\n\nfirst line, edited\nsecond line\n"


async def test_a_missing_passage_is_refused_with_the_next_step(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("alpha\n")

    result = await edit_file(str(path), "beta", "gamma")

    assert result.startswith("Error:")
    assert str(path) in result
    assert "not found" in result
    # The message says what to do, not just what failed.
    assert "read the file again" in result
    assert path.read_text() == "alpha\n"


async def test_an_ambiguous_passage_is_refused_and_names_the_count(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("dup\ndup\n")

    result = await edit_file(str(path), "dup", "other")

    assert "appears 2 times" in result
    assert "replace_all=true" in result
    assert path.read_text() == "dup\ndup\n"


async def test_replace_all_replaces_every_occurrence(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("dup\ndup\ndup\n")

    result = await edit_file(str(path), "dup", "one", replace_all=True)

    assert "replaced 3 occurrences at line 1" in result
    assert path.read_text() == "one\none\none\n"


async def test_an_empty_passage_is_refused(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("alpha\n")

    result = await edit_file(str(path), "", "x")

    assert "old_string is empty" in result
    assert path.read_text() == "alpha\n"


async def test_an_identical_replacement_is_refused(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("alpha\n")

    result = await edit_file(str(path), "alpha", "alpha")

    assert "identical" in result


async def test_a_missing_file_names_write_file(tmp_path):
    path = tmp_path / "absent.md"

    result = await edit_file(str(path), "a", "b")

    assert "does not exist" in result
    assert "write_file" in result


async def test_crlf_line_endings_survive_the_edit(tmp_path):
    path = tmp_path / "windows.txt"
    path.write_bytes(b"one\r\ntwo\r\n")

    result = await edit_file(str(path), "two", "three")

    assert result.startswith("Edited")
    assert path.read_bytes() == b"one\r\nthree\r\n"


async def test_a_non_utf8_file_is_refused_rather_than_mangled(tmp_path):
    path = tmp_path / "blob.bin"
    path.write_bytes(b"\xff\xfe\x00binary")

    result = await edit_file(str(path), "binary", "text")

    assert "not valid UTF-8" in result
    assert path.read_bytes() == b"\xff\xfe\x00binary"


async def test_the_result_reports_the_size_change(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("short\n")

    result = await edit_file(str(path), "short", "a much longer line")

    assert f"file 6 -> {len('a much longer line\n')} bytes" in result


def test_naming_write_file_as_disabled_also_disables_edit_file():
    """An edit tool must not outlive the write tool it stands in for (#1160)."""
    assert expand_disabled(["write_file"]) == frozenset({"write_file", "edit_file"})
    assert expand_disabled(["edit_file"]) == frozenset({"write_file", "edit_file"})


async def test_a_0600_file_stays_0600(tmp_path):
    """The tmp-then-replace dance must not widen a credential file (#1258).

    state/*.env token files are 0600, and a create-then-replace that lets umask
    decide hands back 0644: readable by every user on the box.
    """
    path = tmp_path / "tokens.env"
    path.write_text("TOKEN=old\n")
    path.chmod(0o600)

    result = await edit_file(str(path), "old", "new")

    assert "Edited" in result
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.read_text() == "TOKEN=new\n"


async def test_the_mode_of_an_executable_survives_an_edit(tmp_path):
    path = tmp_path / "run.sh"
    path.write_text("#!/bin/sh\necho old\n")
    path.chmod(0o755)

    await edit_file(str(path), "old", "new")

    assert path.stat().st_mode & 0o777 == 0o755


async def test_editing_through_a_symlink_changes_its_target(tmp_path):
    """os.replace on the link path swaps the link for a file (#1258).

    The edit has to land in the target, and the link has to still be a link;
    otherwise the agent reports success while the real file is untouched.
    """
    target = tmp_path / "real.md"
    target.write_text("before\n")
    link = tmp_path / "link.md"
    link.symlink_to(target)

    result = await edit_file(str(link), "before", "after")

    assert "Edited" in result
    assert target.read_text() == "after\n"
    assert link.is_symlink()
    assert link.read_text() == "after\n"


async def test_no_temporary_file_is_left_behind(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("hello\n")

    await edit_file(str(path), "hello", "goodbye")

    assert [p.name for p in tmp_path.iterdir()] == ["notes.md"]


async def test_parallel_edits_of_one_file_do_not_collide(tmp_path):
    """A fixed ".name.edit-tmp" name lets two edits clobber each other (#1258).

    Each edit may lose the race and report the passage as gone, which is
    honest: what must not happen is a crash, a leftover temp file, or a file
    that is not one of the outcomes of the edits that succeeded.
    """
    path = tmp_path / "notes.md"
    path.write_text("a\nb\nc\n")

    results = await asyncio.gather(
        edit_file(str(path), "a", "A"),
        edit_file(str(path), "b", "B"),
        edit_file(str(path), "c", "C"),
    )

    assert all(r.startswith(("Edited", "Error: old_string was not found")) for r in results)
    assert [p.name for p in tmp_path.iterdir()] == ["notes.md"]
    final = path.read_text()
    assert final.count("a") <= 1 and final.count("b") <= 1 and final.count("c") <= 1
    assert len(final.split()) == 3
