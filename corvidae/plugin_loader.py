"""Plugin loading for the ``corvidae`` entry-point group (#1107).

Replaces pluggy's ``load_setuptools_entrypoints`` plus ``set_blocked`` so a
slot can be filled by a replacement target (``plugins.replace``) and a plugin
that fails to import or construct is skipped instead of stopping startup.

The work is split so the plan is testable without a plugin manager:
``parse_plugins_section`` validates config, ``plan_plugins`` decides what to
load under which name, and ``register_planned`` loads and registers it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint
from typing import Any

logger = logging.getLogger(__name__)

# Keys accepted under ``plugins:`` in agent.yaml.
KNOWN_PLUGINS_KEYS: frozenset[str] = frozenset({"disabled", "replace"})


@dataclass(frozen=True)
class PluginsConfig:
    """The validated ``plugins:`` section."""

    disabled: tuple[str, ...]
    # slot -> target; null, {} or absent mean no replacements.
    replace: Mapping[str, str]


def parse_plugins_section(config: dict) -> PluginsConfig:
    """Validate ``plugins:`` and return it as a PluginsConfig.

    Raises ValueError naming the key, the rejected value and the accepted
    form for anything malformed, including unknown keys under ``plugins:``.
    """
    # A missing or null section configures nothing; otherwise it is a mapping.
    plugins = config.get("plugins")
    if plugins is None:
        return PluginsConfig((), {})
    if not isinstance(plugins, dict):
        raise ValueError(
            f"plugins must be a mapping with `disabled` and `replace` keys, "
            f"got {plugins!r}"
        )
    # An unknown key is a typo that would otherwise silently do nothing.
    unknown = sorted(str(k) for k in plugins if k not in KNOWN_PLUGINS_KEYS)
    if unknown:
        raise ValueError(
            f"plugins has unknown key(s) {', '.join(repr(k) for k in unknown)}; "
            f"accepted keys: {', '.join(sorted(KNOWN_PLUGINS_KEYS))}"
        )
    # disabled: a list of names; null means none.
    disabled = plugins.get("disabled")
    if disabled is None:
        disabled = []
    if not isinstance(disabled, list) or not all(isinstance(n, str) for n in disabled):
        raise ValueError(
            f"plugins.disabled must be a list of plugin names (strings), "
            f"got {disabled!r}"
        )
    # replace: a mapping of slot name to target name; null means none.
    replace = plugins.get("replace")
    if replace is None:
        replace = {}
    if not isinstance(replace, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in replace.items()
    ):
        raise ValueError(
            f"plugins.replace must be a mapping of slot name to replacement "
            f"plugin name (strings), got {replace!r}"
        )
    return PluginsConfig(tuple(disabled), dict(replace))


@dataclass(frozen=True)
class PlanEntry:
    """One plugin to register."""

    # Registered name (the slot's name for a replacement).
    name: str
    # What to load first.
    ep: EntryPoint
    # The bundled entry point when ``ep`` is a replacement target.
    fallback: EntryPoint | None = None


def plan_plugins(
    eps: Sequence[EntryPoint], cfg: PluginsConfig, kernel_names: frozenset[str]
) -> list[PlanEntry]:
    """Decide which entry points load under which names.

    Raises ValueError for config the loader cannot honour (reserved names,
    unknown or kernel slots, disabled or doubly-used targets, chains).
    """
    by_name = {e.name: e for e in eps}
    installed = ", ".join(sorted(by_name))
    slots = set(cfg.replace)

    # A plugin named like a kernel object would be silently skipped by pluggy.
    for name in by_name:
        if name in kernel_names and name not in slots:
            raise ValueError(
                f'entry point "{name}" is reserved for a kernel object; '
                f"rename the entry point in its distribution"
            )

    # Everything not disabled loads, in entry-point order.
    entries: dict[str, PlanEntry] = {
        e.name: PlanEntry(e.name, e) for e in eps if e.name not in cfg.disabled
    }
    used: dict[str, str] = {}  # target -> the slot that took it
    for slot, target in cfg.replace.items():
        if slot in kernel_names:
            choices = ", ".join(sorted(n for n in by_name if not n.startswith("_")))
            raise ValueError(
                f'plugins.replace.{slot}: "{slot}" is a kernel object and cannot '
                f"be replaced; replaceable plugins: {choices}"
            )
        for key, role in ((slot, "slot"), (target, "target")):
            if key not in by_name:
                raise ValueError(
                    f'plugins.replace.{slot}: {role} "{key}" is no installed '
                    f"plugin; installed plugins: {installed}"
                )
            if key in cfg.disabled:
                raise ValueError(
                    f'plugins.replace.{slot}: {role} "{key}" is also in '
                    f"plugins.disabled; remove it from one of them"
                )
        if target == slot:
            raise ValueError(
                f'plugins.replace.{slot}: a plugin cannot replace itself; '
                f"name a different installed plugin or remove the entry"
            )
        if target in slots:
            raise ValueError(
                f'plugins.replace.{slot}: target "{target}" is itself replaced '
                f"(plugins.replace.{target}); chains are not supported, "
                f"name the final plugin directly"
            )
        if target in used:
            raise ValueError(
                f'plugins.replace: target "{target}" is named for more than one '
                f'slot ("{used[target]}" and "{slot}"); a plugin can fill one '
                f"slot, remove one of the entries"
            )
        used[target] = slot
        # The target loads under the slot's name, with the bundled as fallback.
        entries[slot] = PlanEntry(slot, by_name[target], by_name[slot])
        entries.pop(target, None)
    return list(entries.values())


@dataclass
class LoadReport:
    """What the loader did, for the log, corvidae-restart and its preflight."""

    # slot -> target that loaded.
    replaced: dict[str, str] = field(default_factory=dict)
    # slot -> target that failed (the bundled plugin filled the slot).
    fallbacks: dict[str, str] = field(default_factory=dict)
    # Names left unregistered because they failed to load.
    skipped: list[str] = field(default_factory=list)


def _instantiate(ep: EntryPoint) -> Any:
    """Load an entry point and call it if callable (class or factory)."""
    obj = ep.load()
    # A module-valued entry point registers as-is, as pluggy does.
    return obj() if callable(obj) else obj


def _skip(report: LoadReport, name: str, ep_name: str, exc: Exception) -> None:
    """Log an R13 skip for ``ep_name`` and record ``name`` as skipped."""
    logger.error(
        'plugin "%s" failed to load (%s: %s); skipped — fix or reinstall its '
        'distribution, or add "%s" to plugins.disabled',
        ep_name, type(exc).__name__, exc, ep_name, exc_info=True,
    )
    report.skipped.append(name)


def register_planned(pm: Any, plan: list[PlanEntry]) -> LoadReport:
    """Load and register each planned plugin; return what happened.

    A failing import or constructor skips that plugin (or falls back to the
    bundled one for a replacement) and startup continues. A replacement
    target that does not subclass the bundled class raises ValueError.
    """
    report = LoadReport()
    for entry in plan:
        instance: Any = None
        if entry.fallback is None:
            # Ordinary plugin: a load failure skips it.
            try:
                instance = _instantiate(entry.ep)
            except Exception as exc:
                _skip(report, entry.name, entry.ep.name, exc)
                continue
        else:
            slot, target = entry.name, entry.ep.name
            # The bundled class is needed for the subclass check; without it
            # the slot is skipped and the target is not tried.
            try:
                bundled = entry.fallback.load()
            except Exception as exc:
                _skip(report, slot, entry.fallback.name, exc)
                continue
            target_exc: Exception | None = None
            try:
                cls = entry.ep.load()
                # The target must be a subclass of the bundled class.
                if not (isinstance(cls, type) and issubclass(cls, bundled)):
                    raise ValueError(
                        f'plugins.replace.{slot}: "{target}" '
                        f"({getattr(cls, '__name__', repr(cls))}) must subclass "
                        f"{getattr(bundled, '__name__', repr(bundled))}, "
                        f"the plugin it replaces"
                    )
                instance = cls()
                report.replaced[slot] = target
            except ValueError:
                raise
            except Exception as exc:
                target_exc = exc
            if target_exc is not None:
                # Target failed to import or construct: the bundled one fills the slot.
                try:
                    instance = bundled()
                except Exception as bexc:
                    logger.error(
                        'plugin "%s" replacing "%s" failed (%s: %s) and the '
                        'bundled "%s" failed too (%s: %s); neither loaded, '
                        'slot skipped — fix or reinstall them, or add "%s" to '
                        "plugins.disabled",
                        target, slot, type(target_exc).__name__, target_exc,
                        slot, type(bexc).__name__, bexc, slot,
                        exc_info=True,
                    )
                    report.skipped.append(slot)
                    continue
                logger.error(
                    'plugin "%s" replacing "%s" failed to load (%s: %s); using '
                    'the bundled "%s" — fix or reinstall %s, or remove '
                    "plugins.replace.%s",
                    target, slot, type(target_exc).__name__, target_exc,
                    slot, target, slot, exc_info=target_exc,
                )
                report.fallbacks[slot] = target
        # Registration errors are pluggy validation faults and propagate.
        pm.register(instance, name=entry.name)
    return report
