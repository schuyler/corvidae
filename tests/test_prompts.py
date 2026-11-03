"""Tests for corvidae.prompts.resolve_prompt (#411).

Covers: file (dict form), list of files, inline string, absent key,
missing-file error naming the key, _base_dir-relative paths, and
type errors.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from corvidae.prompts import resolve_prompt


@pytest.fixture
def base_dir(tmp_path: Path) -> Path:
    """A temp directory with a couple of prompt files."""
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "a.md").write_text("Alpha prompt.\n")
    (prompts / "b.md").write_text("Beta prompt.\n")
    (prompts / "empty.md").write_text("\n")
    return tmp_path


def _set_dotted(cfg: dict, key: str, value):
    """Set a dotted key in a nested dict: _set_dotted(cfg, 'a.b.c', v)."""
    parts = key.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def _cfg(base: Path, **overrides) -> dict:
    """Build a config dict with _base_dir and dotted-key overrides."""
    cfg: dict = {"_base_dir": base}
    for key, value in overrides.items():
        _set_dotted(cfg, key, value)
    return cfg


class TestInlineString:
    def test_returns_verbatim(self, base_dir):
        cfg = _cfg(base_dir, **{"heartbeat.prompt": "Check the goals."})
        assert resolve_prompt(cfg, "heartbeat.prompt", "DEFAULT") == "Check the goals."

    def test_string_that_names_a_file_is_not_a_path(self, base_dir):
        # A bare string matching a file name is text, not a path.
        (base_dir / "prompts" / "literal.md").write_text("FILE CONTENT")
        cfg = _cfg(base_dir, **{"heartbeat.prompt": "prompts/literal.md"})
        assert resolve_prompt(cfg, "heartbeat.prompt", "DEFAULT") == "prompts/literal.md"

    def test_empty_string_is_a_valid_prompt(self, base_dir):
        cfg = _cfg(base_dir, **{"heartbeat.prompt": ""})
        assert resolve_prompt(cfg, "heartbeat.prompt", "DEFAULT") == ""


class TestDictFile:
    def test_single_file(self, base_dir):
        cfg = _cfg(base_dir, **{"heartbeat.prompt": {"file": "prompts/a.md"}})
        assert resolve_prompt(cfg, "heartbeat.prompt", "DEFAULT") == "Alpha prompt."

    def test_missing_file_raises_with_key_and_path(self, base_dir):
        cfg = _cfg(base_dir, **{"heartbeat.prompt": {"file": "prompts/nope.md"}})
        with pytest.raises(FileNotFoundError, match=r"heartbeat\.prompt.*nope\.md"):
            resolve_prompt(cfg, "heartbeat.prompt", "DEFAULT")

    def test_dict_without_file_key_raises_type_error(self, base_dir):
        cfg = _cfg(base_dir, **{"heartbeat.prompt": {"template": "prompts/a.md"}})
        with pytest.raises(ValueError, match="heartbeat.prompt"):
            resolve_prompt(cfg, "heartbeat.prompt", "DEFAULT")

    def test_absolute_path(self, base_dir):
        cfg = _cfg(base_dir, **{"heartbeat.prompt": {"file": str(base_dir / "prompts" / "b.md")}})
        assert resolve_prompt(cfg, "heartbeat.prompt", "DEFAULT") == "Beta prompt."


class TestFileList:
    def test_two_files_concatenated(self, base_dir):
        cfg = _cfg(base_dir, **{"heartbeat.prompt": ["prompts/a.md", "prompts/b.md"]})
        result = resolve_prompt(cfg, "heartbeat.prompt", "DEFAULT")
        assert result == "Alpha prompt.\n\nBeta prompt."

    def test_empty_list_returns_empty_string(self, base_dir):
        cfg = _cfg(base_dir, **{"heartbeat.prompt": []})
        assert resolve_prompt(cfg, "heartbeat.prompt", "DEFAULT") == ""

    def test_missing_file_in_list_raises(self, base_dir):
        cfg = _cfg(base_dir, **{"heartbeat.prompt": ["prompts/a.md", "prompts/missing.md"]})
        with pytest.raises(FileNotFoundError, match="heartbeat.prompt"):
            resolve_prompt(cfg, "heartbeat.prompt", "DEFAULT")


class TestAbsent:
    def test_missing_key_returns_default(self, base_dir):
        cfg = _cfg(base_dir)
        assert resolve_prompt(cfg, "heartbeat.prompt", "FALLBACK") == "FALLBACK"

    def test_none_value_returns_default(self, base_dir):
        cfg = _cfg(base_dir, **{"heartbeat.prompt": None})
        assert resolve_prompt(cfg, "heartbeat.prompt", "FALLBACK") == "FALLBACK"

    def test_missing_intermediate_dict_returns_default(self, base_dir):
        cfg = _cfg(base_dir)
        assert resolve_prompt(cfg, "nonexistent.nested.key", "FALLBACK") == "FALLBACK"


class TestBaseDir:
    def test_relative_path_resolved_against_base_dir(self, base_dir):
        cfg = _cfg(base_dir, **{"heartbeat.prompt": {"file": "prompts/a.md"}})
        assert resolve_prompt(cfg, "heartbeat.prompt", "DEFAULT") == "Alpha prompt."

    def test_missing_base_dir_falls_back_to_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "prompts").mkdir()
        (tmp_path / "prompts" / "a.md").write_text("From CWD.\n")
        cfg: dict = {"heartbeat": {"prompt": {"file": "prompts/a.md"}}}  # no _base_dir
        assert resolve_prompt(cfg, "heartbeat.prompt", "DEFAULT") == "From CWD."


class TestDottedKeys:
    def test_nested_key(self, base_dir):
        cfg = _cfg(base_dir)
        _set_dotted(cfg, "agent.compaction_fork_prompt", "Summarize the window.")
        assert (
            resolve_prompt(cfg, "agent.compaction_fork_prompt", "DEFAULT")
            == "Summarize the window."
        )

    def test_deeply_nested_file(self, base_dir):
        cfg = _cfg(base_dir)
        _set_dotted(cfg, "critique.fork_prompt.provenance", {"file": "prompts/a.md"})
        assert (
            resolve_prompt(cfg, "critique.fork_prompt.provenance", "DEFAULT")
            == "Alpha prompt."
        )


class TestTypeErrors:
    def test_integer_value_raises(self, base_dir):
        cfg = _cfg(base_dir, **{"heartbeat.prompt": 42})
        with pytest.raises(ValueError, match="heartbeat.prompt"):
            resolve_prompt(cfg, "heartbeat.prompt", "DEFAULT")

    def test_boolean_value_raises(self, base_dir):
        cfg = _cfg(base_dir, **{"heartbeat.prompt": True})
        with pytest.raises(ValueError, match="heartbeat.prompt"):
            resolve_prompt(cfg, "heartbeat.prompt", "DEFAULT")
