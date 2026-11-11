"""A background task's notification is cut at tools.max_result_chars and spilled (#744).

``TaskPlugin._on_task_complete`` sends ``source="task"`` notifications with no
``tool_call_id`` for a task that answers no tool call. The Agent's cut covers
only results that answer a call (#575), so the task plugin cuts its own
notification through the spill service, as PRD §3.3.11 says a sending plugin
does.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from corvidae.channel import Channel, ChannelConfig
from corvidae.hooks import create_plugin_manager, hookimpl
from corvidae.spill import NotificationSpill, current_cut
from corvidae.task import Task, TaskPlugin
from corvidae.tool import Tool
from corvidae.tool_collection import ToolCollectionPlugin
from fake_clock import install_kernel_clock

LIMIT = 40
# "[Task <id>] " is prepended to the result, so the cut counts it too.
BIG = "x" * 200


class Recorder:
    """Collects what on_notify delivers, with the cut the spill service set."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    @hookimpl
    async def on_notify(self, channel, source, text, tool_call_id, annotations):
        self.calls.append(
            {"text": text, "source": source, "tool_call_id": tool_call_id,
             "annotations": annotations, "cut": current_cut()}
        )


class Readers:
    """A tool plugin whose one tool declares it reads spill files."""

    @hookimpl
    def register_tools(self, tool_registry: list) -> None:
        async def read_file(path: str) -> str:
            """Stub tool."""
            return ""

        read_file.reads_spill_files = True  # type: ignore[attr-defined]
        tool_registry.append(Tool.from_function(read_file))


@pytest.fixture
async def rig(tmp_path):
    """A plugin manager with the task plugin, spill service and a recorder."""
    clocks = []

    async def build(*, spill: bool = True, limit: int = LIMIT, spill_dir: Path | None = None):
        pm = create_plugin_manager()
        clock, _ = install_kernel_clock(pm, started=False)
        clocks.append(clock)
        directory = spill_dir or tmp_path / "spill"
        config = {"_base_dir": tmp_path, "tools": {"max_result_chars": limit},
                  "daemon": {"notification_spill_dir": str(directory)}}
        collection = ToolCollectionPlugin(pm)
        pm.register(collection, name="tools")
        pm.register(Readers(), name="readers")
        recorder = Recorder()
        pm.register(recorder, name="recorder")
        if spill:
            service = NotificationSpill()
            pm.register(service, name="spill")
            await service.on_init(pm=pm, config=config)
        await collection.on_init(pm=pm, config=config)
        await collection.rebuild_registry()
        plugin = TaskPlugin(pm)
        await plugin.on_start(config={})
        return pm, plugin, collection, recorder, directory

    yield build
    for clock in clocks:
        await clock.close()


def make_task(*, tool_call_id: str | None = None) -> Task:
    async def work() -> str:
        return ""

    return Task(
        work=work, channel=Channel(transport="test", scope="s", config=ChannelConfig()),
        tool_call_id=tool_call_id, description="background work",
    )


async def test_long_background_result_is_cut_and_spilled(rig):
    """Over the cap: the notification is cut, and the file holds the full text."""
    pm, plugin, _, recorder, directory = await rig()
    task = make_task()
    await plugin._on_task_complete(task, BIG)

    [call] = recorder.calls
    full = f"[Task {task.task_id}] {BIG}"
    assert call["text"] == full[:LIMIT]
    assert call["source"] == "task"
    assert call["tool_call_id"] is None
    assert call["annotations"]["task_id"] == task.task_id
    cut = call["cut"]
    assert (cut.shown, cut.total) == (LIMIT, len(full))
    assert Path(cut.path).parent == directory.resolve()
    assert Path(cut.path).read_text(encoding="utf-8") == full
    await plugin.on_stop()


async def test_result_within_the_cap_is_unchanged(rig):
    """Under the cap: delivered whole, nothing cut, nothing written."""
    pm, plugin, _, recorder, directory = await rig(limit=1000)
    task = make_task()
    await plugin._on_task_complete(task, "short")

    [call] = recorder.calls
    assert call["text"] == f"[Task {task.task_id}] short"
    assert call["cut"] is None
    assert not directory.exists() or not list(directory.iterdir())
    await plugin.on_stop()


async def test_a_result_that_answers_a_tool_call_is_left_to_the_agent(rig):
    """With a tool_call_id the Agent cuts it (#575); the plugin must not cut twice."""
    pm, plugin, _, recorder, _ = await rig()
    task = make_task(tool_call_id="call_1")
    await plugin._on_task_complete(task, BIG)

    [call] = recorder.calls
    assert call["text"] == f"[Task {task.task_id}] {BIG}"
    assert call["tool_call_id"] == "call_1"
    assert call["cut"] is None
    await plugin.on_stop()


async def test_an_empty_call_id_is_no_call_id(rig):
    """The Agent treats "" as no id, so the plugin cuts the result as it would for None."""
    pm, plugin, _, recorder, _ = await rig()
    task = make_task(tool_call_id="")
    await plugin._on_task_complete(task, BIG)

    [call] = recorder.calls
    assert len(call["text"]) == LIMIT
    assert call["cut"] is not None
    await plugin.on_stop()


async def test_the_cap_follows_a_live_change_of_max_result_chars(rig):
    """The cap is read at delivery, so a settings change applies to the next result."""
    pm, plugin, collection, recorder, _ = await rig(limit=1000)
    task = make_task()
    await plugin._on_task_complete(task, BIG)
    collection.max_result_chars = 60
    await plugin._on_task_complete(task, BIG)

    assert len(recorder.calls[0]["text"]) == len(f"[Task {task.task_id}] {BIG}")
    assert len(recorder.calls[1]["text"]) == 60
    await plugin.on_stop()


async def test_an_unwritable_spill_directory_still_delivers_an_excerpt(rig, tmp_path, caplog):
    """The spill cannot be written: the excerpt goes out, the cut says it was not
    saved, and a warning names the failure."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("file in the way")
    pm, plugin, _, recorder, _ = await rig(spill_dir=blocker / "spill")
    task = make_task()
    with caplog.at_level(logging.WARNING, logger="corvidae.spill"):
        await plugin._on_task_complete(task, BIG)

    [call] = recorder.calls
    assert len(call["text"]) == LIMIT
    assert call["cut"].failed is True
    assert call["cut"].path is None
    assert "not saved" in caplog.text
    await plugin.on_stop()


async def test_without_a_spill_service_the_result_is_delivered_whole_with_a_warning(rig, caplog):
    """A bare plugin manager (tests, embedders) keeps today's behavior and says so."""
    pm, plugin, _, recorder, _ = await rig(spill=False)
    task = make_task()
    with caplog.at_level(logging.WARNING, logger="corvidae.task"):
        await plugin._on_task_complete(task, BIG)

    [call] = recorder.calls
    assert call["text"] == f"[Task {task.task_id}] {BIG}"
    assert "spill" in caplog.text and task.task_id in caplog.text
    # The message names the cap and the remedy, not just the failure.
    assert "tools.max_result_chars (40)" in caplog.text
    assert "register NotificationSpill" in caplog.text
    await plugin.on_stop()
