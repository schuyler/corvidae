"""edit_file: an exact-passage edit that refuses when it cannot be sure (#1160)."""

from __future__ import annotations

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
