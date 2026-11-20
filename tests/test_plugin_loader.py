"""plugins.replace and the plugin loader (#1107).

Unit tests drive ``corvidae.plugin_loader`` (parse, plan, register) against
fake entry points and a real plugin manager; two Runtime tests drive the
"plugins loaded" record (R9) and the reserved ``_plugin_load`` config key.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from corvidae.hooks import create_plugin_manager, get_dependency
from tests import plugin_loader_fixtures as fx
from tests.plugin_loader_fixtures import ep

# Module path the fake entry points resolve through.
FIX = "tests.plugin_loader_fixtures"

# The kernel objects Runtime registers before loading entry points.
KERNEL = frozenset({"registry", "clock", "idle", "spill", "_seed_hooks"})


# Entry points shared by most tests: a bundled "llm" slot and a good target.
BUNDLED_LLM = ep("llm", "Bundled")
MY_LLM = ep("my_llm", "GoodTarget")


@pytest.fixture(autouse=True)
def _reset_calls() -> None:
    """Start every test with no recorded constructor calls."""
    fx.CALLS.clear()


def load(eps, plugins: dict | None = None, kernel=KERNEL):
    """Parse a plugins section, plan, and register into a real manager."""
    from corvidae import plugin_loader as pl

    cfg = pl.parse_plugins_section({"plugins": plugins})
    plan = pl.plan_plugins(eps, cfg, kernel)
    pm = create_plugin_manager()
    report = pl.register_planned(pm, plan)
    return pm, report


def errors(caplog) -> list[str]:
    """Messages of every ERROR record."""
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]


# --- R1 / R2 ------------------------------------------------------------------


def test_r1_target_takes_the_slot() -> None:
    """R1: get_plugin and get_dependency on the slot return the target."""
    pm, report = load([BUNDLED_LLM, MY_LLM], {"replace": {"llm": "my_llm"}})
    assert type(pm.get_plugin("llm")) is fx.GoodTarget
    assert get_dependency(pm, "llm", fx.Bundled) is pm.get_plugin("llm")
    assert report.replaced == {"llm": "my_llm"}


def test_r2_bundled_never_instantiated_and_target_not_under_own_name() -> None:
    """R2: the bundled constructor never runs; no registration as my_llm."""
    pm, _ = load([BUNDLED_LLM, MY_LLM], {"replace": {"llm": "my_llm"}})
    assert fx.CALLS["Bundled"] == 0
    assert fx.CALLS["GoodTarget"] == 1
    assert pm.get_plugin("my_llm") is None


# --- R3 -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "target_ep, target_cls",
    [
        (ep("my_llm", "NotASub"), "NotASub"),
        # A module-valued target is not a class either (fix-now item).
        (ep("my_llm", FIX), None),
    ],
)
def test_r3_target_must_subclass_bundled(target_ep, target_cls) -> None:
    """R3: a non-subclass target is a ValueError naming both and the key."""
    with pytest.raises(ValueError) as excinfo:
        load([BUNDLED_LLM, target_ep], {"replace": {"llm": "my_llm"}})
    msg = str(excinfo.value)
    assert "plugins.replace.llm" in msg and "my_llm" in msg and "Bundled" in msg
    if target_cls:
        assert target_cls in msg


# --- R4 / R5 ------------------------------------------------------------------


def test_r4_kernel_object_slot_is_rejected() -> None:
    """R4: a kernel object cannot be replaced; the message names it and the choices."""
    with pytest.raises(ValueError) as excinfo:
        load([BUNDLED_LLM, MY_LLM], {"replace": {"clock": "my_llm"}})
    msg = str(excinfo.value)
    assert "clock" in msg and "plugins.replace" in msg
    # The accepted slots are installed entry points; seed plumbing is not offered.
    assert "llm" in msg and "_seed_hooks" not in msg


def test_r5_unknown_slot_lists_installed_names() -> None:
    """R5: an unknown slot names it and lists installed entry points."""
    with pytest.raises(ValueError) as excinfo:
        # "unrelated" is installed but not in the mapping: only a real list of
        # installed names can put it in the message.
        load(
            [BUNDLED_LLM, MY_LLM, ep("unrelated", "Plain")],
            {"replace": {"nope": "my_llm"}},
        )
    msg = str(excinfo.value)
    assert "nope" in msg and "llm" in msg and "my_llm" in msg
    assert "unrelated" in msg


def test_r5_unknown_target_lists_installed_names() -> None:
    """R5: an unknown target names it and lists installed entry points."""
    with pytest.raises(ValueError) as excinfo:
        load(
            [BUNDLED_LLM, MY_LLM, ep("unrelated", "Plain")],
            {"replace": {"llm": "ghost"}},
        )
    msg = str(excinfo.value)
    assert "ghost" in msg and "llm" in msg and "my_llm" in msg
    assert "unrelated" in msg


# --- R6 -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "target_ep, exc_type",
    [
        (ep("my_llm", "tests.plugin_loader_boom:Anything"), "ImportError"),
        (ep("my_llm", "CtorBoom"), "RuntimeError"),
    ],
)
def test_r6_failing_target_falls_back_to_bundled(caplog, target_ep, exc_type) -> None:
    """R6: import or constructor failure in the target; bundled fills the slot."""
    with caplog.at_level(logging.ERROR):
        pm, report = load([BUNDLED_LLM, target_ep], {"replace": {"llm": "my_llm"}})
    # The bundled plugin serves the slot and the report records the fallback.
    assert type(pm.get_plugin("llm")) is fx.Bundled
    assert report.fallbacks == {"llm": "my_llm"}
    assert report.replaced == {} and report.skipped == []
    # One ERROR names slot, target, exception type and the remedy.
    msgs = errors(caplog)
    assert len(msgs) == 1, msgs
    for needle in ("llm", "my_llm", exc_type, "plugins.replace.llm"):
        assert needle in msgs[0], msgs[0]


def test_both_target_and_bundled_failing_skips_the_slot(caplog) -> None:
    """Fix-now: neither loads; slot is skipped, not a fallback; one ERROR."""
    bundled = ep("llm", "BundledCtorBoom")
    target = ep("my_llm", "CtorBoomSub")
    with caplog.at_level(logging.ERROR):
        pm, report = load([bundled, target], {"replace": {"llm": "my_llm"}})
    assert pm.get_plugin("llm") is None
    assert report.skipped == ["llm"] and report.fallbacks == {}
    msgs = errors(caplog)
    assert len(msgs) == 1, msgs
    assert "neither" in msgs[0] and "llm" in msgs[0] and "my_llm" in msgs[0]
    assert "boom-target-ctor" in msgs[0] or "RuntimeError" in msgs[0]


# --- R7 / R8 and parsing --------------------------------------------------------


def test_r7_slot_in_disabled_is_rejected() -> None:
    """R7: a slot that is also disabled stops startup, naming it."""
    with pytest.raises(ValueError) as excinfo:
        load(
            [BUNDLED_LLM, MY_LLM],
            {"disabled": ["llm"], "replace": {"llm": "my_llm"}},
        )
    assert "llm" in str(excinfo.value) and "disabled" in str(excinfo.value)


def test_r7_target_in_disabled_is_rejected() -> None:
    """R7: a target that is also disabled stops startup, naming it."""
    with pytest.raises(ValueError) as excinfo:
        load(
            [BUNDLED_LLM, MY_LLM],
            {"disabled": ["my_llm"], "replace": {"llm": "my_llm"}},
        )
    assert "my_llm" in str(excinfo.value) and "disabled" in str(excinfo.value)


def test_r7_one_target_in_two_slots_is_rejected() -> None:
    """R7: one target cannot fill two slots.

    Only ``plan_plugins`` runs: the planning check, not the R3 subclass check
    (GoodTarget is not an OtherBundled), is what must reject this.
    """
    from corvidae import plugin_loader as pl

    eps = [BUNDLED_LLM, ep("other", "OtherBundled"), ep("my_llm", "GoodTarget")]
    cfg = pl.parse_plugins_section(
        {"plugins": {"replace": {"llm": "my_llm", "other": "my_llm"}}}
    )
    with pytest.raises(ValueError) as excinfo:
        pl.plan_plugins(eps, cfg, KERNEL)
    msg = str(excinfo.value)
    assert "my_llm" in msg and "more than one slot" in msg, msg
    assert "llm" in msg and "other" in msg, msg
    assert "plugins.replace" in msg, msg


@pytest.mark.parametrize(
    "replace",
    [["llm"], "llm", {"llm": 3}, {3: "my_llm"}, {"llm": ["my_llm"]}],
)
def test_r7_replace_must_map_strings_to_strings(replace) -> None:
    """R7: a malformed replace value names plugins.replace and the value."""
    from corvidae import plugin_loader as pl

    with pytest.raises(ValueError) as excinfo:
        pl.parse_plugins_section({"plugins": {"replace": replace}})
    msg = str(excinfo.value)
    assert "plugins.replace" in msg and repr(replace) in msg


def test_r8_unknown_plugins_key_names_key_and_accepted_keys() -> None:
    """R8: an unknown key under plugins: names it and the accepted keys."""
    from corvidae import plugin_loader as pl

    with pytest.raises(ValueError) as excinfo:
        pl.parse_plugins_section({"plugins": {"disabld": ["x"]}})
    msg = str(excinfo.value)
    assert "disabld" in msg and "disabled" in msg and "replace" in msg


@pytest.mark.parametrize("replace", [None, {}])
def test_null_or_empty_replace_means_none(replace) -> None:
    """Fix-now: ``replace: null`` and ``{}`` are no replacements."""
    from corvidae import plugin_loader as pl

    cfg = pl.parse_plugins_section({"plugins": {"replace": replace, "disabled": ["a"]}})
    assert dict(cfg.replace) == {} and cfg.disabled == ("a",)


@pytest.mark.parametrize(
    "replace, needle",
    [
        ({"llm": "my_llm", "my_llm": "extra"}, "is itself replaced"),
        ({"llm": "llm"}, "cannot replace itself"),
    ],
)
def test_chain_and_self_replacement_are_rejected(replace, needle) -> None:
    """Fix-now: a chain is order-dependent and ``llm: llm`` is meaningless.

    Only ``plan_plugins`` runs, so the R3 subclass check cannot rescue the
    chain case: its own wording has to be produced by the planning check.
    """
    from corvidae import plugin_loader as pl

    eps = [BUNDLED_LLM, MY_LLM, ep("extra", "Plain")]
    cfg = pl.parse_plugins_section({"plugins": {"replace": replace}})
    with pytest.raises(ValueError) as excinfo:
        pl.plan_plugins(eps, cfg, KERNEL)
    assert needle in str(excinfo.value) and "llm" in str(excinfo.value)


def test_kernel_named_entry_point_is_rejected() -> None:
    """Fix-now: an entry point named like a kernel object is a Config error."""
    with pytest.raises(ValueError) as excinfo:
        load([ep("registry", "Plain")], None)
    assert "registry" in str(excinfo.value) and "reserved" in str(excinfo.value)


def test_module_valued_entry_point_registers_as_is() -> None:
    """Fix-now: a non-callable entry-point object registers unchanged."""
    import types

    pm, report = load([ep("modplug", FIX)], None)
    assert isinstance(pm.get_plugin("modplug"), types.ModuleType)
    assert report.skipped == []


# --- R9 -----------------------------------------------------------------------


def _runtime_config(tmp_path: Path, plugins: dict) -> tuple[dict, str]:
    """A minimal config dict plus a config file path for Runtime."""
    config = {
        "llm": {
            "providers": {"main": {"base_url": "http://localhost:8080"}},
            "profiles": {"main": {"model": "m"}},
        },
        "plugins": plugins,
        "_base_dir": str(tmp_path),
    }
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.safe_dump({k: v for k, v in config.items() if k[0] != "_"}))
    return config, str(path)


async def _runtime_load(tmp_path: Path, monkeypatch, eps, plugins: dict):
    """Run Runtime.load_plugins over fake entry points; return (runtime, config)."""
    from corvidae.runtime import Runtime

    config, path = _runtime_config(tmp_path, plugins)
    monkeypatch.setattr("corvidae.runtime.entry_points", fx.entry_points_for(*eps))
    monkeypatch.setattr("corvidae.runtime.check_entry_point_duplicates", lambda g="corvidae": None)
    rt = Runtime(config_path=path)
    await rt.load_plugins(config)
    return rt, config


async def test_r9_plugins_loaded_record_carries_replaced(tmp_path, monkeypatch, caplog) -> None:
    """R9: the "plugins loaded" record's extra lists swaps (and no fallbacks)."""
    with caplog.at_level(logging.INFO, logger="corvidae.runtime"):
        await _runtime_load(
            tmp_path, monkeypatch, [BUNDLED_LLM, MY_LLM],
            {"replace": {"llm": "my_llm"}},
        )
    record = next(r for r in caplog.records if r.getMessage() == "plugins loaded")
    assert dict(record.replaced) == {"llm": "my_llm"}
    assert dict(record.fallbacks) == {}


async def test_r9_plugins_loaded_record_carries_fallbacks(tmp_path, monkeypatch, caplog) -> None:
    """R9: a failed target shows under fallbacks, not replaced."""
    with caplog.at_level(logging.INFO, logger="corvidae.runtime"):
        await _runtime_load(
            tmp_path, monkeypatch, [BUNDLED_LLM, ep("my_llm", "CtorBoom")],
            {"replace": {"llm": "my_llm"}},
        )
    record = next(r for r in caplog.records if r.getMessage() == "plugins loaded")
    assert dict(record.replaced) == {}
    assert dict(record.fallbacks) == {"llm": "my_llm"}


async def test_load_report_reaches_plugins_through_reserved_config_key(
    tmp_path, monkeypatch
) -> None:
    """R11 feed: config["_plugin_load"] is the runtime's LoadReport at on_init."""
    fx.CAPTURED.clear()
    rt, config = await _runtime_load(
        tmp_path,
        monkeypatch,
        [BUNDLED_LLM, MY_LLM, ep("capture", "InitCapture")],
        {"replace": {"llm": "my_llm"}},
    )
    # The key was present when the plugin's own on_init ran, not just after.
    assert fx.CAPTURED.get("report") is rt.load_report, fx.CAPTURED
    assert config["_plugin_load"] is rt.load_report
    assert rt.load_report.replaced == {"llm": "my_llm"}


async def test_unknown_replace_target_exits_with_config_error(
    tmp_path, monkeypatch, capsys
) -> None:
    """R5 at Runtime level: a bad plugins.replace value is one Config error line."""
    with pytest.raises(SystemExit) as excinfo:
        await _runtime_load(
            tmp_path, monkeypatch, [BUNDLED_LLM, MY_LLM],
            {"replace": {"llm": "ghost"}},
        )
    assert excinfo.value.code == 1
    stderr = capsys.readouterr().err
    assert "Config error:" in stderr, stderr
    assert "plugins.replace" in stderr and "ghost" in stderr, stderr


# --- R10 ----------------------------------------------------------------------


def test_r10_changed_plugins_replace_logs_restart_only_warning(tmp_path, caplog) -> None:
    """R10 (guard; passes before Green — no code change is needed for R10).

    ``ConfigWatcherPlugin`` warns for every changed key no plugin claims as
    reloadable, and nothing claims ``plugins``.
    """
    from corvidae.config_watcher import ConfigWatcherPlugin

    running = {"plugins": {"replace": {"llm": "a"}}}
    cfg = tmp_path / "agent.yaml"
    cfg.write_text(yaml.safe_dump(running))
    pm = create_plugin_manager()
    watcher = ConfigWatcherPlugin()
    pm.register(watcher, name="config_watcher")
    watcher.pm = pm
    watcher._config_path = cfg
    watcher._cli_overrides = {}
    watcher.config = {"_config_path": str(cfg), **running}

    with caplog.at_level(logging.WARNING):
        watcher._warn_restart_only_changes(
            {"_config_path": str(cfg), "plugins": {"replace": {"llm": "b"}}}
        )
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("plugins.replace.llm" in m and "restart-only" in m for m in msgs), msgs


# --- R13 ----------------------------------------------------------------------


def test_r13_failing_plugins_are_skipped_and_others_register(caplog) -> None:
    """R13: import and constructor failures skip that plugin only, with an ERROR."""
    eps = [
        ep("good_a", "Plain"),
        ep("bad_import", "tests.plugin_loader_boom:Anything"),
        ep("bad_ctor", "CtorBoom"),
        ep("good_b", "Plain2"),
    ]
    with caplog.at_level(logging.ERROR):
        pm, report = load(eps, None)
    assert sorted(report.skipped) == ["bad_ctor", "bad_import"]
    assert pm.get_plugin("good_a") is not None and pm.get_plugin("good_b") is not None
    assert pm.get_plugin("bad_import") is None and pm.get_plugin("bad_ctor") is None
    msgs = errors(caplog)
    by_name = {n: m for n in ("bad_import", "bad_ctor") for m in msgs if f'"{n}"' in m}
    assert set(by_name) == {"bad_import", "bad_ctor"}, msgs
    for name, exc_type in (("bad_import", "ImportError"), ("bad_ctor", "RuntimeError")):
        assert exc_type in by_name[name], by_name[name]
        assert "plugins.disabled" in by_name[name], by_name[name]
