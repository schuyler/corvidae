"""The restart-only warning reflects what a restart would actually change (#473).

Changing a restart-only key logs "changed; restart-only, takes effect on
restart". Editing that key back to the value the daemon is running with must
silence the warning: the running config and the file agree, so a restart
would change nothing.
"""

import logging

import pytest
import yaml

from corvidae.compaction import CompactionPlugin
from corvidae.config_watcher import ConfigWatcherPlugin
from corvidae.hooks import create_plugin_manager


RUNNING = {
    "llm": {"main": {"base_url": "http://localhost:8080/v1", "model": "startup"}},
}


def _watcher(tmp_path, running: dict) -> ConfigWatcherPlugin:
    """A watcher whose running config is ``running``.

    Compaction is registered because `agent.*` is a key prefix it declares
    live, which the live-key test relies on.
    """
    cfg = tmp_path / "agent.yaml"
    cfg.write_text(yaml.safe_dump(running))
    pm = create_plugin_manager()
    watcher = ConfigWatcherPlugin()
    pm.register(watcher, name="config_watcher")
    pm.register(CompactionPlugin(), name="compaction")
    watcher.pm = pm
    watcher._config_path = cfg
    watcher._cli_overrides = {}
    watcher.config = {"_config_path": str(cfg), **running}
    return watcher


def _new(**overrides) -> dict:
    """A reloaded config: the running one with ``llm.main.model`` replaced."""
    return {
        "_config_path": "agent.yaml",
        "llm": {"main": {
            "base_url": "http://localhost:8080/v1",
            "model": overrides.get("model", "startup"),
        }},
    }


def _warnings(caplog) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
    ]


def test_reverting_a_key_to_the_running_value_does_not_warn(tmp_path, caplog):
    """R473.1: reload 1 changes the model, reload 2 sets it back.

    Reload 2 must log no restart-only warning for that key: the running
    config already holds the value the file names.
    """
    watcher = _watcher(tmp_path, RUNNING)

    with caplog.at_level(logging.WARNING):
        watcher._warn_restart_only_changes(_new(model="changed"))
        assert [m for m in _warnings(caplog) if "llm.main.model" in m], (
            "reload 1 must report the change"
        )

        caplog.clear()
        watcher._warn_restart_only_changes(_new(model="startup"))
        reverted = [m for m in _warnings(caplog) if "llm.main.model" in m]

    assert not reverted, (
        f"a key reverted to its running value must not warn; got {reverted}"
    )


def test_a_pending_key_is_not_re_reported_while_it_stays_pending(tmp_path, caplog):
    """R473.2 (owner choice): one report per pending value, not per reload."""
    watcher = _watcher(tmp_path, RUNNING)

    with caplog.at_level(logging.WARNING):
        watcher._warn_restart_only_changes(_new(model="changed"))
        caplog.clear()
        watcher._warn_restart_only_changes(_new(model="changed"))
        repeated = [m for m in _warnings(caplog) if "llm.main.model" in m]

    assert not repeated, f"a still-pending key must not be re-reported: {repeated}"


def test_a_new_pending_value_is_reported_again(tmp_path, caplog):
    """A second, different edit is news and is reported."""
    watcher = _watcher(tmp_path, RUNNING)

    with caplog.at_level(logging.WARNING):
        watcher._warn_restart_only_changes(_new(model="first"))
        caplog.clear()
        watcher._warn_restart_only_changes(_new(model="second"))
        again = [m for m in _warnings(caplog) if "llm.main.model" in m]

    assert again, "a changed pending value must be reported"


def test_a_reverted_key_can_warn_again_after_a_new_edit(tmp_path, caplog):
    """A revert clears the pending state, so a later edit warns again."""
    watcher = _watcher(tmp_path, RUNNING)

    with caplog.at_level(logging.WARNING):
        watcher._warn_restart_only_changes(_new(model="changed"))
        watcher._warn_restart_only_changes(_new(model="startup"))  # reverted
        caplog.clear()
        watcher._warn_restart_only_changes(_new(model="third"))
        again = [m for m in _warnings(caplog) if "llm.main.model" in m]

    assert again, "after a revert, a fresh edit must be reported"


def test_a_key_removed_back_to_absence_does_not_warn(tmp_path, caplog):
    """Deleting a key that the running config does not have is not a change."""
    watcher = _watcher(tmp_path, RUNNING)

    with caplog.at_level(logging.WARNING):
        watcher._warn_restart_only_changes(
            {"_config_path": "agent.yaml", **RUNNING}
        )
        messages = [m for m in _warnings(caplog) if "llm.main.model" in m]

    assert not messages, f"an unchanged file must not warn: {messages}"


def test_live_keys_still_do_not_warn(tmp_path, caplog):
    """A key a plugin reloads is applied at once and is never reported."""
    running = {**RUNNING, "agent": {"compaction_threshold": 0.5}}
    watcher = _watcher(tmp_path, running)

    with caplog.at_level(logging.WARNING):
        watcher._warn_restart_only_changes(
            {**running, "agent": {"compaction_threshold": 0.9}}
        )
        messages = [m for m in _warnings(caplog) if "compaction_threshold" in m]

    assert not messages, f"a live key must not warn: {messages}"
