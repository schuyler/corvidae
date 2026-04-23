"""Tests for sherman.prompt.resolve_system_prompt."""

from pathlib import Path

import pytest

from sherman.prompt import resolve_system_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DUMMY_BASE_DIR = Path("/tmp/irrelevant")


# ---------------------------------------------------------------------------
# String passthrough
# ---------------------------------------------------------------------------


class TestStringPassthrough:
    def test_string_passthrough(self):
        """String input is returned unchanged; base_dir is not consulted."""
        result = resolve_system_prompt("You are a helpful assistant.", DUMMY_BASE_DIR)
        assert result == "You are a helpful assistant."

    def test_empty_string_passthrough(self):
        """Empty string is a valid string value and passes through unchanged."""
        result = resolve_system_prompt("", DUMMY_BASE_DIR)
        assert result == ""


# ---------------------------------------------------------------------------
# List inputs
# ---------------------------------------------------------------------------


class TestListInputs:
    def test_list_single_file(self, tmp_path):
        """List with one path reads and returns that file's content."""
        f = tmp_path / "prompt.md"
        f.write_text("You are a bot.")

        result = resolve_system_prompt([str(f)], tmp_path)
        assert result == "You are a bot."

    def test_list_multiple_files(self, tmp_path):
        """List with multiple paths concatenates content with double newlines."""
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        a.write_text("Part one.")
        b.write_text("Part two.")

        result = resolve_system_prompt([str(a), str(b)], tmp_path)
        assert result == "Part one.\n\nPart two."

    def test_list_strips_whitespace(self, tmp_path):
        """Leading and trailing whitespace in each file's content is stripped."""
        f = tmp_path / "padded.md"
        f.write_text("  \n  Content here.  \n  ")

        result = resolve_system_prompt([str(f)], tmp_path)
        assert result == "Content here."

    def test_list_strips_whitespace_multiple_files(self, tmp_path):
        """Whitespace stripping applies per-file before concatenation."""
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        a.write_text("  First.  ")
        b.write_text("\nSecond.\n")

        result = resolve_system_prompt([str(a), str(b)], tmp_path)
        assert result == "First.\n\nSecond."

    def test_empty_list_returns_empty_string(self, tmp_path):
        """Empty list produces an empty string."""
        result = resolve_system_prompt([], tmp_path)
        assert result == ""


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


class TestPathResolution:
    def test_relative_path_resolved_against_base_dir(self, tmp_path):
        """Relative path in list is resolved against base_dir, not cwd."""
        f = tmp_path / "soul.md"
        f.write_text("Identity prompt.")

        # Pass a relative name, not the absolute path.
        result = resolve_system_prompt(["soul.md"], tmp_path)
        assert result == "Identity prompt."

    def test_absolute_path_used_directly(self, tmp_path):
        """Absolute path is used as-is, regardless of base_dir."""
        f = tmp_path / "absolute.md"
        f.write_text("Absolute prompt.")

        # base_dir is a different (non-existent) location — should not matter.
        fake_base = Path("/nonexistent/dir")
        result = resolve_system_prompt([str(f.resolve())], fake_base)
        assert result == "Absolute prompt."


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestErrorCases:
    def test_missing_file_raises(self, tmp_path):
        """FileNotFoundError is raised when a path in the list does not exist."""
        with pytest.raises(FileNotFoundError):
            resolve_system_prompt(["does_not_exist.md"], tmp_path)

    def test_invalid_type_raises(self):
        """Non-str, non-list value raises TypeError."""
        with pytest.raises(TypeError):
            resolve_system_prompt(42, DUMMY_BASE_DIR)  # type: ignore[arg-type]

    def test_invalid_type_dict_raises(self):
        """Dict value raises TypeError."""
        with pytest.raises(TypeError):
            resolve_system_prompt({"key": "val"}, DUMMY_BASE_DIR)  # type: ignore[arg-type]

    def test_invalid_type_none_raises(self):
        """None raises TypeError (callers should filter None before calling)."""
        with pytest.raises(TypeError):
            resolve_system_prompt(None, DUMMY_BASE_DIR)  # type: ignore[arg-type]
