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


def test_a_key_removed_from_the_file_is_reported(tmp_path, caplog):
    """R473.1: deleting a restart-only key changes what a restart produces.

    An absent value and the "never reported" default must be distinguishable
    sentinels; conflating them silently suppressed this warning entirely.
    """
    watcher = _watcher(tmp_path, RUNNING)

    with caplog.at_level(logging.WARNING):
        # The reloaded file no longer names `model` at all.
        watcher._warn_restart_only_changes(
            {"_config_path": "agent.yaml",
             "llm": {"main": {"base_url": "http://localhost:8080/v1"}}}
        )
        messages = [m for m in _warnings(caplog) if "llm.main.model" in m]

    assert messages, "deleting a restart-only key must be reported"


def test_a_key_name_containing_a_dot_is_reported(tmp_path, caplog):
    """A profile name may contain a dot (harness models are `qwen3.6:35b`).

    The changed path is walked by key name, not by splitting a dotted string,
    so `llm.qwen3.6.model` resolves and every change to it is reported.
    Splitting it made the lookup read `_ABSENT` each time, which still lets
    the first change through, so one reload cannot tell the two apart: the
    second and third reloads are the ones the bug silenced (#473).
    """
    running = {"llm": {"qwen3.6": {"base_url": "http://h:1", "model": "a"}}}
    watcher = _watcher(tmp_path, running)

    def reload_with(model: str) -> dict:
        return {"_config_path": "agent.yaml",
                "llm": {"qwen3.6": {"base_url": "http://h:1", "model": model}}}

    reported = []
    with caplog.at_level(logging.WARNING):
        for model in ("b", "c", "d"):
            caplog.clear()
            watcher._warn_restart_only_changes(reload_with(model))
            reported.append([m for m in _warnings(caplog) if "llm.qwen3.6.model" in m])

    assert [bool(r) for r in reported] == [True, True, True], (
        f"every change to a dotted profile name must be reported; got {reported}"
    )


def test_an_unchanged_file_does_not_warn(tmp_path, caplog):
    """A reload that changes nothing draws no warning.

    The reloaded config is the running one and nothing else: a fixture that
    added a key would produce a real warning for it, and filtering on
    `llm.main.model` would then hide that warning and report green (review).
    """
    watcher = _watcher(tmp_path, RUNNING)

    with caplog.at_level(logging.WARNING):
        watcher._warn_restart_only_changes({"_config_path": "agent.yaml", **RUNNING})
        messages = _warnings(caplog)

    assert not messages, f"an unchanged file must not warn: {messages}"


def test_a_non_string_key_reports_every_change(tmp_path, caplog):
    """A key YAML gives a non-string type (an int) still reports each change.

    The path is walked by the key's real name. Coercing it to a string first
    made `_leaf_value` miss the key and return `_ABSENT` every time, so only
    the first change was reported and later ones compared equal to it.
    """
    running = {"llm": {1: {"base_url": "http://h:1", "model": "a"}}}
    watcher = _watcher(tmp_path, running)

    def reload_with(model: str) -> dict:
        return {"_config_path": "agent.yaml",
                "llm": {1: {"base_url": "http://h:1", "model": model}}}

    reported = []
    with caplog.at_level(logging.WARNING):
        for model in ("b", "c", "d"):
            caplog.clear()
            watcher._warn_restart_only_changes(reload_with(model))
            reported.append(bool(_warnings(caplog)))

    assert reported == [True, True, True], (
        f"every change to a non-string key must be reported; got {reported}"
    )


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


# ---------------------------------------------------------------------------
# R473.3: the two observers agree
#
# The ticket requires that a revert produce no restart-only line *and* no
# `llm config changed` line. The second half is LLMPlugin's, and it was never
# covered by a test even though its behaviour is correct (#473 review).
# ---------------------------------------------------------------------------


def _llm_plugin_with(running_llm: dict) -> object:
    """An LLMPlugin whose parsed running config is ``running_llm``."""
    from corvidae.llm_plugin import LLMPlugin, parse_llm_config

    plugin = LLMPlugin()
    plugin._config = parse_llm_config(running_llm)
    return plugin


async def test_a_revert_produces_no_llm_config_changed_line(caplog):
    """R473.3: a revert logs neither observer's line.

    The scenario is the ticket's: reload 1 changes the model, reload 2 sets it
    back. A single reload of an unchanged config would not distinguish this
    from the "nothing changed" case, and would pass even if `LLMPlugin` diffed
    against the previous reload -- the bug #473 exists to prevent (review).
    """
    running = {"main": {"base_url": "http://localhost:8080/v1", "model": "startup"}}
    plugin = _llm_plugin_with(running)
    changed = {"main": {"base_url": "http://localhost:8080/v1", "model": "other"}}

    with caplog.at_level(logging.WARNING):
        # Reload 1: a real change, which must be reported.
        await plugin.on_config_reload({"llm": changed})
        assert any("llm config changed" in m for m in _warnings(caplog)), (
            "reload 1 must report the change"
        )

        # Reload 2: back to the running config, which must not.
        caplog.clear()
        await plugin.on_config_reload({"llm": dict(running)})

    assert not _warnings(caplog), (
        f"a revert must not log: {_warnings(caplog)}"
    )


async def test_a_real_llm_change_still_produces_the_summary_line(caplog):
    """The summary line is not silenced, only made truthful on a revert."""
    running = {"main": {"base_url": "http://localhost:8080/v1", "model": "startup"}}
    plugin = _llm_plugin_with(running)
    changed = {"main": {"base_url": "http://localhost:8080/v1", "model": "other"}}

    with caplog.at_level(logging.WARNING):
        await plugin.on_config_reload({"llm": changed})

    assert any("llm config changed" in m for m in _warnings(caplog)), (
        "a real llm change must still be reported"
    )



# #591: the live-prefix filter compares key segments, not a dotted string.
class _LiveAgentStub:
    """Declares the whole ``agent`` section live, as the issue's criteria assume."""

    reloadable_keys = ("agent",)


def _reload(**top_level) -> dict:
    return {"_config_path": "agent.yaml", **top_level}


def _warned(watcher, caplog, config, needle) -> bool:
    with caplog.at_level(logging.WARNING):
        watcher._warn_restart_only_changes(config)
    return any(needle in m for m in _warnings(caplog))


def test_a_dotted_key_name_is_not_a_live_prefix_child(tmp_path, caplog):
    """A top-level key *named* ``agent.forks`` is not under the live prefix ``agent``."""
    watcher = _watcher(tmp_path, RUNNING)
    watcher.pm.register(_LiveAgentStub(), name="live_agent_stub")
    assert _warned(watcher, caplog, _reload(**{"agent.forks": True}), "agent.forks")


def test_a_nested_key_under_a_live_prefix_is_still_filtered(tmp_path, caplog):
    watcher = _watcher(tmp_path, RUNNING)
    watcher.pm.register(_LiveAgentStub(), name="live_agent_stub")
    config = _reload(agent={"compaction_threshold": 0.5})
    assert not _warned(watcher, caplog, config, "compaction_threshold")
