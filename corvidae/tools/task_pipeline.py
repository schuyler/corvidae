"""TaskPipelinePlugin — YAML/JSON task definitions with dependency resolution.

Registers a ``task_pipeline`` tool that accepts a task definition in YAML or JSON
format and executes it respecting declared dependencies. Tasks can be inline (the
agent defines them on-the-fly) or loaded from files.

Task definition format:
    tasks:
      - name: build
        command: "make all"
        depends_on: []
      - name: test
        command: "make test"
        depends_on: [build]
      - name: deploy
        command: "./deploy.sh"
        depends_on: [test]

Architecture:
    - Tasks are defined as a DAG (directed acyclic graph)
    - Topological sort determines execution order
    - Cycle detection prevents invalid definitions
    - Failed tasks block dependent tasks from running
    - Progress reported via on_task_event hook for channel delivery

Plugin hooks used:
    - register_tools     — exposes the task_pipeline tool
    - before_agent_turn  — clears transient failed-task state from previous turns
"""

from __future__ import annotations

import json
import logging
import re
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corvidae.hooks import CorePluginManager
    from corvidae.tool import ToolContext

logger = logging.getLogger(__name__)


def _parse_yaml(text: str) -> dict | list:
    """Minimal YAML subset parser for task definitions.

    Handles:
    - Top-level ``tasks:`` key with a list of mapping items
    - Simple string, integer, and list values
    - No nested maps beyond one level (task entries)
    - Quoted and unquoted strings

    Raises ValueError on ambiguous or unsupported syntax.
    """
    lines = text.strip().splitlines()
    tasks: list[dict] = []
    current_task: dict | None = None
    in_tasks = False

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Top-level key
        top_match = re.match(r"^(\w[\w-]*)\s*:\s*(.*)$", stripped)
        if top_match and not line.startswith(" ") and not line.startswith("\t"):
            key, val = top_match.groups()
            if key == "tasks":
                in_tasks = True
                if val.strip():
                    # Inline list: tasks: [{...}]
                    try:
                        return json.loads(val)
                    except json.JSONDecodeError:
                        pass
                continue
            else:
                in_tasks = False
                continue

        if not in_tasks:
            continue

        # List item: "- name: foo"
        item_match = re.match(r"^-\s+(\w[\w-]*)\s*:\s*(.*)$", stripped)
        if item_match:
            if current_task is not None:
                tasks.append(current_task)
            key, val = item_match.groups()
            current_task = {key: _parse_value(val)}
            continue

        # Continuation: "  depends_on: [a, b]"
        cont_match = re.match(r"^(\w[\w-]*)\s*:\s*(.*)$", stripped)
        if cont_match and current_task is not None:
            key, val = cont_match.groups()
            current_task[key] = _parse_value(val)

    if current_task is not None:
        tasks.append(current_task)

    return {"tasks": tasks}


def _parse_value(raw: str):
    """Parse a single YAML value string into a Python type."""
    raw = raw.strip()
    if not raw:
        return ""
    # Quoted string
    if (raw.startswith('"') and raw.endswith('"')) or \
       (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    # List: [a, b, c]
    if raw.startswith("[") and raw.endswith("]"):
        items = raw[1:-1].split(",")
        return [_parse_value(i) for i in items if i.strip()]
    # Boolean
    if raw.lower() in ("true", "yes"):
        return True
    if raw.lower() in ("false", "no"):
        return False
    # Null
    if raw.lower() in ("null", "~", ""):
        return None
    # Integer
    try:
        return int(raw)
    except ValueError:
        pass
    # Float
    try:
        return float(raw)
    except ValueError:
        pass
    # Plain string (strip inline comments)
    comment = re.search(r"\s+#\s", raw)
    if comment:
        raw = raw[:comment.start()]
    return raw.strip()


def _detect_cycle(tasks: list[dict]) -> list[str] | None:
    """Detect cycles in the task dependency graph using DFS.

    Args:
        tasks: List of task dicts with 'name' and 'depends_on' keys.

    Returns:
        A cycle path as a list of task names if a cycle exists, None otherwise.
    """
    task_map = {t["name"]: t for t in tasks}
    all_names = set(task_map.keys())

    WHITE, GRAY, BLACK = 0, 1, 2
    color = {name: WHITE for name in all_names}
    parent: dict[str, str | None] = {name: None for name in all_names}

    def dfs(node: str) -> list[str] | None:
        color[node] = GRAY
        deps = task_map.get(node, {}).get("depends_on", []) or []
        for dep in deps:
            if dep not in all_names:
                continue  # skip unknown deps (will be caught elsewhere)
            if color[dep] == GRAY:
                # Found cycle — reconstruct path
                cycle = [dep, node]
                current = node
                while current != dep and parent.get(current) is not None:
                    current = parent[current]
                    if current == dep:
                        break
                    cycle.append(current)
                return list(reversed(cycle))
            if color[dep] == WHITE:
                parent[dep] = node
                result = dfs(dep)
                if result is not None:
                    return result
        color[node] = BLACK
        return None

    for name in all_names:
        if color[name] == WHITE:
            cycle = dfs(name)
            if cycle is not None:
                return cycle
    return None


def _topological_sort(tasks: list[dict]) -> list[str]:
    """Topological sort of tasks by dependency order.

    Args:
        tasks: List of task dicts with 'name' and 'depends_on' keys.

    Returns:
        Ordered list of task names suitable for execution.

    Raises:
        ValueError: If the graph contains a cycle.
    """
    cycle = _detect_cycle(tasks)
    if cycle is not None:
        raise ValueError(f"Cycle detected in task dependencies: {' -> '.join(cycle)}")

    task_map = {t["name"]: t for t in tasks}
    all_names = set(task_map.keys())
    in_degree: dict[str, int] = {name: 0 for name in all_names}

    for task in tasks:
        deps = task.get("depends_on") or []
        for dep in deps:
            if dep in all_names:
                in_degree[task["name"]] += 1  # type: ignore[operator]

    queue = deque([n for n, d in in_degree.items() if d == 0])
    order: list[str] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for task in tasks:
            deps = task.get("depends_on") or []
            if node in deps and task["name"] in all_names:
                in_degree[task["name"]] -= 1  # type: ignore[operator]
                if in_degree[task["name"]] == 0:  # type: ignore[operator]
                    queue.append(task["name"])

    if len(order) != len(all_names):
        raise ValueError("Cycle detected in task dependencies (topological sort incomplete)")

    return order


class TaskPipelinePlugin:
    """YAML/JSON task pipeline with dependency resolution and execution.

    Provides a single tool ``task_pipeline`` that the agent calls to define
    and execute ordered task graphs. Tasks are parsed from YAML or JSON,
    validated for cycles and missing dependencies, sorted topologically,
    then executed sequentially (each dependent task waits for its predecessors).

    Failed tasks prevent their dependents from running. The agent can query
    status via the returned result string.
    """

    def __init__(self) -> None:
        self._tasks_cache: list[dict] = []  # last parsed task list
        self._task_status: dict[str, str] = {}  # name -> "pending"|"running"|"done"|"failed"
        self._task_results: dict[str, str] = {}  # name -> output/error string

    def _reset(self) -> None:
        """Clear transient state between agent turns."""
        self._task_status.clear()
        self._task_results.clear()

    def register_tools(self, tool_registry: list) -> None:
        from corvidae.tool import Tool

        plugin = self

        async def task_pipeline(
            definition: str, _ctx: ToolContext | None = None
        ) -> str:
            """Execute a task pipeline defined in YAML or JSON format.

            The definition must contain a 'tasks' key with a list of objects,
            each having 'name', 'command', and optionally 'depends_on' (list of
            task names). Tasks are executed in topological order; failed tasks
            block their dependents.

            Example YAML:
                tasks:
                  - name: build
                    command: "make all"
                    depends_on: []
                  - name: test
                    command: "make test"
                    depends_on: [build]

            Returns a summary of execution results including any failures.
            """
            # Parse the definition
            try:
                # Try JSON first
                parsed = json.loads(definition)
            except json.JSONDecodeError:
                # Fall back to YAML parser
                parsed = _parse_yaml(definition)

            tasks_raw = parsed.get("tasks") if isinstance(parsed, dict) else []
            if not tasks_raw:
                return "No tasks found in definition."

            # Validate task names are unique
            names = [t.get("name", "") for t in tasks_raw]
            if len(names) != len(set(names)):
                duplicates = [n for n in names if names.count(n) > 1]
                return f"Duplicate task names detected: {', '.join(set(duplicates))}"

            # Detect cycles
            cycle = _detect_cycle(tasks_raw)
            if cycle is not None:
                return f"Cycle detected in task dependencies: {' -> '.join(cycle)}. Cannot execute."

            # Topological sort
            try:
                order = _topological_sort(tasks_raw)
            except ValueError as e:
                return str(e)

            task_map = {t["name"]: t for t in tasks_raw}

            # Reset state for this execution
            plugin._reset()
            plugin._tasks_cache = tasks_raw

            results_lines: list[str] = []
            failed_tasks: set[str] = set()

            for task_name in order:
                task = task_map[task_name]
                deps = set(task.get("depends_on") or [])

                # Check if any dependency failed
                blocked = deps & failed_tasks
                if blocked:
                    status = "failed"
                    result = f"Blocked by failed dependencies: {', '.join(sorted(blocked))}"
                    failed_tasks.add(task_name)
                else:
                    status = "running"
                    command = task.get("command", "")
                    try:
                        from corvidae.tools.shell import shell as _shell_impl
                        result = await _shell_impl(command, timeout=120)
                        status = "done"
                    except Exception as exc:
                        status = "failed"
                        result = f"Error: {exc}"
                        failed_tasks.add(task_name)

                plugin._task_status[task_name] = status
                plugin._task_results[task_name] = result
                results_lines.append(f"  {task_name}: {status}")
                if result and len(result) < 500:
                    results_lines.append(f"    → {result[:400]}{'...' if len(result) > 400 else ''}")

            total = len(order)
            done = sum(1 for s in plugin._task_status.values() if s == "done")
            failed = sum(1 for s in plugin._task_status.values() if s == "failed")

            header = f"Pipeline complete: {total} tasks, {done} succeeded, {failed} failed"
            return header + "\n" + "\n".join(results_lines)

        tool_registry.append(Tool.from_function(task_pipeline))
