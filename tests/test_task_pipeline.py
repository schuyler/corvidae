"""Tests for corvidae.tools.task_pipeline."""

import pytest

from corvidae.tools.task_pipeline import (
    _detect_cycle,
    _parse_value,
    _parse_yaml,
    _topological_sort,
)


# --- YAML parsing tests ---


class TestParseYaml:
    def test_basic_tasks(self):
        yaml_def = """
tasks:
  - name: build
    command: "make all"
    depends_on: []
  - name: test
    command: "make test"
    depends_on: [build]
"""
        result = _parse_yaml(yaml_def)
        assert isinstance(result, dict)
        assert "tasks" in result
        assert len(result["tasks"]) == 2
        assert result["tasks"][0]["name"] == "build"
        assert result["tasks"][1]["depends_on"] == ["build"]

    def test_empty_tasks(self):
        yaml_def = "tasks:\n  - name: a\n    command: echo hi\n    depends_on: []"
        result = _parse_yaml(yaml_def)
        assert len(result["tasks"]) == 1

    def test_inline_list_value(self):
        yaml_def = """
tasks:
  - name: deploy
    depends_on: [build, test]
"""
        result = _parse_yaml(yaml_def)
        assert result["tasks"][0]["depends_on"] == ["build", "test"]

    def test_non_task_yaml_returns_empty(self):
        """YAML without a 'tasks' key returns empty tasks list."""
        yaml_def = "other_key: value"
        result = _parse_yaml(yaml_def)
        assert result == {"tasks": []}


class TestParseValue:
    def test_quoted_string(self):
        assert _parse_value('"hello world"') == "hello world"
        assert _parse_value("'single quotes'") == "single quotes"

    def test_integer(self):
        assert _parse_value("42") == 42

    def test_boolean_true(self):
        assert _parse_value("true") is True

    def test_boolean_false(self):
        assert _parse_value("false") is False

    def test_null(self):
        assert _parse_value("null") is None
        assert _parse_value("~") is None

    def test_list(self):
        assert _parse_value("[a, b, c]") == ["a", "b", "c"]

    def test_plain_string(self):
        assert _parse_value("hello world") == "hello world"
        assert _parse_value("hello world # comment") == "hello world"


# --- Cycle detection tests ---


class TestDetectCycle:
    def test_no_cycle(self):
        tasks = [
            {"name": "a", "depends_on": []},
            {"name": "b", "depends_on": ["a"]},
            {"name": "c", "depends_on": ["b"]},
        ]
        assert _detect_cycle(tasks) is None

    def test_direct_cycle(self):
        tasks = [
            {"name": "a", "depends_on": ["b"]},
            {"name": "b", "depends_on": ["a"]},
        ]
        assert _detect_cycle(tasks) is not None

    def test_three_node_cycle(self):
        tasks = [
            {"name": "a", "depends_on": ["c"]},
            {"name": "b", "depends_on": ["a"]},
            {"name": "c", "depends_on": ["b"]},
        ]
        assert _detect_cycle(tasks) is not None

    def test_no_cycle_with_unknown_dep(self):
        tasks = [
            {"name": "a", "depends_on": ["missing"]},
            {"name": "b", "depends_on": ["a"]},
        ]
        assert _detect_cycle(tasks) is None


# --- Topological sort tests ---


class TestTopologicalSort:
    def test_linear(self):
        tasks = [
            {"name": "a", "depends_on": []},
            {"name": "b", "depends_on": ["a"]},
            {"name": "c", "depends_on": ["b"]},
        ]
        order = _topological_sort(tasks)
        assert order == ["a", "b", "c"]

    def test_diamond(self):
        tasks = [
            {"name": "a", "depends_on": []},
            {"name": "b", "depends_on": ["a"]},
            {"name": "c", "depends_on": ["a"]},
            {"name": "d", "depends_on": ["b", "c"]},
        ]
        order = _topological_sort(tasks)
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_raises_on_cycle(self):
        tasks = [
            {"name": "a", "depends_on": ["b"]},
            {"name": "b", "depends_on": ["a"]},
        ]
        with pytest.raises(ValueError, match="Cycle"):
            _topological_sort(tasks)

    def test_single_task(self):
        tasks = [{"name": "solo", "depends_on": []}]
        assert _topological_sort(tasks) == ["solo"]
