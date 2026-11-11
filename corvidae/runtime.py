"""Agent lifecycle management for corvidae subcommands.

The Runtime class extracts the agent startup/shutdown lifecycle from the
legacy main() function into a reusable class that any subcommand can
import and use.

Shutdown:
    SIGINT/SIGTERM trigger graceful shutdown via stop_event. A second signal
    during shutdown triggers os._exit(1) immediately. Graceful shutdown runs
    under ``daemon.shutdown_timeout`` (default 3 s); os._exit(1) on timeout.
    Entry points call ``run_daemon``, which also arms an exit watchdog:
    ``daemon.exit_timeout`` (default 5 s, 0/null disables) after the graceful
    phase ends or raises, a process still alive (a leaked non-daemon thread)
    is forced out with status 1 and a WARNING naming the threads.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from importlib.metadata import entry_points
from pathlib import Path

import yaml

from corvidae.channel import ChannelRegistry, load_channel_config, validate_agent_config
from corvidae.clock import Clock
from corvidae.hooks import (
    _check_hook_arg_binding,
    create_plugin_manager,
    install_plugin_hookspecs,
    report_orphan_hookimpls,
    validate_dependencies,
)
from corvidae.idle import IdleTracker, validate_daemon_config
from corvidae.llm_plugin import parse_llm_config
from corvidae.logging import configure_logging
from corvidae.shutdown import (
    DEFAULT_SHUTDOWN_TIMEOUTS,
    arm_exit_watchdog,
    parse_shutdown_timeouts,
)
from corvidae.spill import NotificationSpill
from corvidae.tool_collection import parse_read_file_window_fraction, parse_tool_timeouts
from corvidae.tools.subagent import parse_subagent_max_turns

logger = logging.getLogger(__name__)


def check_entry_point_duplicates(group: str = "corvidae") -> None:
    """Fail loudly when two distributions claim one entry-point name (#238 R6).

    pluggy's load_setuptools_entrypoints skips a name it has already
    registered, so an install-order winner would otherwise be silent. The
    error names the entry point, both distributions, and the remedy.
    """
    import importlib.metadata

    by_name: dict[str, list] = {}
    for ep in importlib.metadata.entry_points(group=group):
        by_name.setdefault(ep.name, []).append(ep)
    for name, eps in sorted(by_name.items()):
        if len(eps) < 2:
            continue
        dists = sorted({ep.dist.name for ep in eps if ep.dist is not None})
        raise ValueError(
            f"duplicate '{group}' entry point '{name}' provided by "
            f"{', '.join(dists) if dists else 'unknown distributions'}; "
            f"uninstall one of them"
        )


def _warn_unknown_disabled(disabled: list[str], registered: set[str]) -> None:
    """Warn once for each plugins.disabled entry that names no plugin.

    Such an entry blocks nothing (a typo, a retired name such as the old
    combined tools plugin, or a plugin whose distribution is not installed),
    so it gets a WARNING naming the key, the value and the plugin names it
    could have meant, beyond the INFO "plugins loaded" line. Known names are
    every ``corvidae`` entry point from every installed distribution plus the
    plugins registered directly before blocking; only the entry-point names
    are offered as choices, since the kernel objects are not meant to be
    disabled.
    """
    entry_point_names = {ep.name for ep in entry_points(group="corvidae")}
    known = entry_point_names | registered
    # dict.fromkeys dedupes while keeping the operator's order.
    for name in dict.fromkeys(disabled):
        if name not in known:
            logger.warning(
                'plugins.disabled names "%s", which is no installed plugin; '
                "it blocks nothing. Installed plugins: %s",
                name, ", ".join(sorted(entry_point_names)),
            )


def disabled_plugins_from(config: dict) -> list[str]:
    """Return ``plugins.disabled`` as a list of names, validated.

    A missing or null section, or a null ``disabled``, means nothing is
    disabled. Raises ValueError naming the key, the rejected value and the
    accepted form for anything else, so a bare string is never iterated
    letter by letter into a block list.
    """
    # A missing or null section disables nothing; it must otherwise be a mapping.
    plugins = config.get("plugins")
    if plugins is None:
        return []
    if not isinstance(plugins, dict):
        raise ValueError(
            f"plugins must be a mapping with a `disabled` list of plugin "
            f"names, got {plugins!r}"
        )
    # A null disabled list is the same as none; anything else must be a
    # list of names.
    disabled = plugins.get("disabled")
    if disabled is None:
        return []
    if not isinstance(disabled, list) or not all(
        isinstance(name, str) for name in disabled
    ):
        raise ValueError(
            f"plugins.disabled must be a list of plugin names (strings), "
            f"got {disabled!r}"
        )
    return disabled


def deep_merge(base: dict, overrides: dict) -> dict:
    """Deep-merge overrides on top of base, returning a new dict.

    Merge rules:
    - Dicts merge recursively.
    - Non-dict values replace the base value.
    - None values in overrides are skipped (base value preserved).
    - Top-level keys absent from base are added.
    - The base argument is not mutated.
    """
    result = dict(base)
    for key, override_val in overrides.items():
        if override_val is None:
            # None is a no-op: preserve whatever base has (or its absence).
            continue
        base_val = result.get(key)
        if isinstance(base_val, dict) and isinstance(override_val, dict):
            result[key] = deep_merge(base_val, override_val)
        else:
            result[key] = override_val
    return result


class Runtime:
    """Manages the plugin manager lifecycle for corvidae subcommands."""

    def __init__(
        self,
        config_path: str = "agent.yaml",
        overrides: dict | None = None,
    ):
        self.config_path = config_path
        self.overrides = overrides or {}
        self.pm = None
        self.registry = None
        self.clock = None
        self.idle = None
        # Shutdown bounds; replaced by the parsed daemon section at startup.
        self.shutdown_timeouts = DEFAULT_SHUTDOWN_TIMEOUTS

    def load_config(self) -> dict:
        """Steps 1-3 of start(): read the YAML, merge overrides, set reserved keys.

        Prints one Config error line and raises SystemExit(1) if the file is
        missing, undecodable, malformed or not a mapping. Returns the config.
        """
        # 1. Load YAML config, reporting a missing or malformed file as a
        # one-line operator error instead of an uncaught traceback.
        # The bytes are decoded explicitly, so the file is read as UTF-8
        # whatever the locale and a decode error's offset counts from the
        # start of the file (a text stream reports it per 8 KiB chunk).
        try:
            text = Path(self.config_path).read_bytes().decode("utf-8")
            config = yaml.safe_load(text)
        except UnicodeDecodeError as exc:
            # Not an OSError or a YAMLError: name it, with the fix (#310).
            sys.stderr.write(
                f"Config error: file is not valid UTF-8 ({exc.reason} at byte "
                f"{exc.start}); re-save it as UTF-8 — check {self.config_path}\n"
            )
            sys.stderr.flush()
            raise SystemExit(1) from exc
        except RecursionError as exc:
            # PyYAML's composer recurses per nesting level (#310).
            sys.stderr.write(
                "Config error: YAML is nested too deeply to load; reduce the "
                f"nesting — check {self.config_path}\n"
            )
            sys.stderr.flush()
            raise SystemExit(1) from exc
        except (OSError, yaml.YAMLError) as exc:
            detail = getattr(exc, "strerror", None) or str(exc)
            sys.stderr.write(f"Config error: {detail} — check {self.config_path}\n")
            sys.stderr.flush()
            raise SystemExit(1) from exc
        # An empty or scalar config cannot be merged, so treat it as malformed.
        if not isinstance(config, dict):
            sys.stderr.write(
                f"Config error: file is empty or not a mapping — check {self.config_path}\n"
            )
            sys.stderr.flush()
            raise SystemExit(1)

        # 2. Deep-merge overrides
        config = deep_merge(config, self.overrides)

        # 3. Set reserved keys after merge — cannot be overridden via YAML.
        # _base_dir: parent directory of config_path for resolving relative paths.
        # _config_path: Path to the config file, used by ConfigWatcherPlugin to poll for changes.
        # _cli_overrides: CLI overrides dict, re-applied on every config reload.
        config["_base_dir"] = Path(self.config_path).parent
        config["_config_path"] = Path(self.config_path)
        config["_cli_overrides"] = self.overrides

        return config

    def setup_logging(self, config: dict) -> None:
        """Step 4 of start(): configure logging from the config's logging section."""
        # 4. Configure logging — must be first operational step. Relative
        # paths resolve against _base_dir, same mechanism as
        # MetricsJsonlPlugin's daemon.metrics_jsonl.
        log_section = config.get("logging", {})
        log_level = log_section.get("level", "INFO")
        log_file = log_section.get("file")
        if log_file is not None:
            log_file = str(config["_base_dir"] / log_file)
        configure_logging(level=log_level, file=log_file)
        logger.info(
            "logging configured",
            extra={"level": log_level, "file": log_file or "stderr"},
        )

    async def load_plugins(self, config: dict) -> None:
        """Steps 5-10 of start(): build the plugin manager, load and init plugins.

        Prints one Config error line and raises SystemExit(1) for a rejected
        config section, a duplicate entry point or an on_init ValueError.
        """
        # 5. Create plugin manager
        self.pm = create_plugin_manager()

        # 6. Create ChannelRegistry, Clock, IdleTracker and NotificationSpill and register them.
        # Kernel objects: not loaded from entry points, not disableable.
        self.registry = ChannelRegistry()
        self.pm.register(self.registry, name="registry")
        self.clock = Clock()
        self.idle = IdleTracker(self.clock)
        self.pm.register(self.clock, name="clock")
        self.pm.register(self.idle, name="idle")
        self.spill = NotificationSpill()
        self.pm.register(self.spill, name="spill")

        # 7. Validate the agent section and load channel config. A rejected
        # value is an operator error: one line naming the key, the value
        # and the file, like a malformed file above, not a traceback.
        try:
            # daemon.shutdown_timeout / exit_timeout: parsed first so a later
            # failure here still cleans up under the configured bound.
            self.shutdown_timeouts = parse_shutdown_timeouts(config.get("daemon"))
            validate_agent_config(config.get("agent", {}))
            validate_daemon_config(config.get("daemon") or {})
            # The llm profiles, defaults and stage map: a bad block exits
            # here, before any plugin loads.
            parse_llm_config(config.get("llm"))
            disabled_plugins = disabled_plugins_from(config)
            # tools.call_timeout / tools.timeouts: reject a bad value at startup.
            parse_tool_timeouts(config.get("tools"))
            # tools.read_file_window_fraction: reject a bad value at startup.
            parse_read_file_window_fraction(config.get("tools"))
            # subagent.max_turns: reject a non-positive-integer budget at
            # startup (int(False) == 0 would no-op every subagent silently).
            parse_subagent_max_turns(config)
            self.registry.agent_defaults = config.get("agent", {})
            load_channel_config(config, self.registry)
        except ValueError as exc:
            sys.stderr.write(f"Config error: {exc} — check {self.config_path}\n")
            sys.stderr.flush()
            raise SystemExit(1) from exc

        # 7b. Block disabled plugins before loading entry points so they are
        #     never instantiated. Must precede load_setuptools_entrypoints.
        #     Snapshot the directly registered names first: set_blocked
        #     unregisters a plugin already registered under the name, and
        #     step 8a must still count it as a real plugin name.
        registered_before_blocking = {
            name for name, plugin in self.pm.list_name_plugin() if plugin is not None
        }
        for name in disabled_plugins:
            self.pm.set_blocked(name)

        # 8. Load entry-point plugins. A duplicate name is an operator
        #    error (two distributions shipping the same plugin): one line
        #    and exit, like a malformed config, not a silent install-order
        #    winner (#238 R6).
        try:
            check_entry_point_duplicates("corvidae")
        except ValueError as exc:
            sys.stderr.write(f"Config error: {exc}\n")
            sys.stderr.flush()
            raise SystemExit(1) from exc
        self.pm.load_setuptools_entrypoints("corvidae")

        # 8a. Record the effective plugin set: what actually loaded and what
        # the operator asked to disable, verbatim (a typo must still show up
        # here even though the plugin it was meant to block still loaded).
        # pluggy keeps each blocked name as (name, None); those did not load.
        loaded_names = sorted(
            name for name, plugin in self.pm.list_name_plugin()
            if plugin is not None
        )
        logger.info(
            "plugins loaded",
            extra={"loaded": loaded_names, "blocked": disabled_plugins},
        )
        _warn_unknown_disabled(disabled_plugins, registered_before_blocking)

        # 8b. Install the hookspecs plugins declare (#229). Every plugin is
        #     registered by now, and specs must exist before the guard
        #     below checks impls against them and before any on_init calls
        #     another plugin's hook.
        install_plugin_hookspecs(self.pm)

        # 8c. Guard the full, real plugin set against pluggy's silent-drop
        #     arg-binding bug class. create_plugin_manager() only checks the
        #     seed hooks; the real entry-point plugins are not registered
        #     until the line above, so this second check covers them.
        _check_hook_arg_binding(self.pm)

        # 9. Validate dependencies, then report hookimpls with no spec: a
        #    typo, or an owner that isn't loaded.
        validate_dependencies(self.pm)
        report_orphan_hookimpls(self.pm)

        # 10. on_init broadcast. A plugin that rejects its config raises
        #     ValueError here (the IRC, Discord and mcp convention): report
        #     it as the one-line operator error step 7 uses for a malformed
        #     block, and exit — not as a traceback (#277). Any other
        #     exception is a real fault and still raises.
        try:
            await self.pm.ahook.on_init(pm=self.pm, config=config)
        except ValueError as exc:
            sys.stderr.write(f"Config error: {exc} — check {self.config_path}\n")
            sys.stderr.flush()
            raise SystemExit(1) from exc

    async def start(self) -> None:
        """Load config, merge overrides, create PM, load plugins, start.

        Delegates steps 1-3 to load_config(), step 4 to setup_logging() and
        steps 5-10 to load_plugins(), then runs steps 11-13 itself.

        Steps:
        1.  Load YAML config from config_path. Prints a Config error and exits
            non-zero if missing or malformed.
        2.  Deep-merge self.overrides on top of the YAML config.
        3.  Set config["_base_dir"] from config_path (after merge; cannot be overridden).
        4.  Configure logging (MUST be first operational step after config load).
        5.  Create plugin manager.
        6.  Create ChannelRegistry, Clock, IdleTracker and NotificationSpill and
            register them with the plugin manager as "registry", "clock",
            "idle" and "spill".
        7.  Validate the agent, daemon and llm sections and tools.call_timeout /
            tools.timeouts; load channel config from merged config.
        7b. Block disabled plugins via pm.set_blocked for each name in
            config["plugins"]["disabled"]. Must precede step 8 so blocked
            plugins are never instantiated by the entry-point loader.
        8.  Load entry-point plugins (corvidae group).
        8a. Log the effective plugin set, and warn for each disabled name that names no plugin.
        8b. Install plugin-declared hookspecs (install_plugin_hookspecs), so
            every spec exists before the guard below and before on_init.
        8c. Run the hook arg-binding guard over the full plugin set.
        9.  Validate plugin dependencies, then report hookimpls left with no
            spec (report_orphan_hookimpls).
        10. await pm.ahook.on_init(pm=pm, config=config). A ValueError from
            a plugin's on_init is a config rejection: one-line Config error
            and exit, like a malformed block in step 7 (#277).
        11. await pm.ahook.on_start(config=config).
            NOTE: Agent.on_start has no @hookimpl — the broadcast skips it.
        12. Retrieve agent plugin; raise RuntimeError if absent.
            await agent.on_start(config=config) explicitly, after the broadcast
            so that plugins Agent depends on have already started.
        13. Start the idle tracker, so nothing fires before the Agent can
            take work.
        """
        # 1-3. Load, merge and mark the config.
        config = self.load_config()

        # 4. Configure logging.
        self.setup_logging(config)

        # 5-10. Build the plugin manager, load plugins, run on_init.
        await self.load_plugins(config)

        # 11. on_start broadcast (Agent.on_start has no @hookimpl — skipped here)
        await self.pm.ahook.on_start(config=config)

        # 12. Explicit agent.on_start — after broadcast so dependencies have started
        agent = self.pm.get_plugin("agent")
        if agent is None:
            raise RuntimeError("Agent plugin not registered — check entry points")
        await agent.on_start(config=config)

        # 13. Start the idle tracker now that the Agent can take work.
        self.idle.start()

    async def stop(self) -> None:
        """Graceful shutdown: agent.on_stop, pm.ahook.on_stop, idle.stop, clock.close.

        Runs under ``daemon.shutdown_timeout`` (default 3 s); os._exit(1) on
        timeout. The exit watchdog (``daemon.exit_timeout``) is armed by
        run(), not here. Raises RuntimeError if agent plugin is not registered.
        """
        agent = self.pm.get_plugin("agent")
        if agent is None:
            raise RuntimeError("Agent plugin not registered — cannot stop cleanly")

        try:
            await asyncio.wait_for(
                self._run_shutdown(agent),
                timeout=self.shutdown_timeouts.graceful,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "graceful shutdown timed out after %ss (daemon.shutdown_timeout), "
                "force-exiting",
                self.shutdown_timeouts.graceful,
            )
            logging.shutdown()
            os._exit(1)

    async def _run_shutdown(self, agent: object) -> None:
        """Run agent and plugin shutdown in order. Called under a timeout in stop()."""
        await agent.on_stop()
        await self.pm.ahook.on_stop()
        # Stop the tracker, then cancel any timers plugins left on the clock.
        self.idle.stop()
        await self.clock.close()

    async def _cleanup_failed_start(self) -> None:
        """Best-effort shutdown after start() raised (#177).

        Also the teardown for the restart pre-flight, which loads plugins
        without starting an agent.

        Runs the on_stop broadcast, stops the idle tracker and closes the
        clock under the same bound as stop() (daemon.shutdown_timeout).
        Skips agent.on_stop, since the agent may never have started. Every failure here is logged
        and swallowed, so the start failure is what the caller sees.
        """
        # Config loading failed before a plugin manager existed: nothing to close.
        if self.pm is None:
            return
        # Same steps and bound as stop(), minus agent.on_stop. clock and idle
        # exist whenever pm does (start() creates all three together).
        try:
            # Wake any message a transport delivered before the Agent
            # started, so its on_message returns instead of hanging (#71).
            agent = self.pm.get_plugin("agent")
            if agent is not None:
                agent.release_held_messages()
            async with asyncio.timeout(self.shutdown_timeouts.graceful):
                await self.pm.ahook.on_stop()
                self.idle.stop()
                await self.clock.close()
        except TimeoutError:
            logger.warning(
                "cleanup after failed startup timed out after %ss "
                "(daemon.shutdown_timeout)",
                self.shutdown_timeouts.graceful,
            )
        except Exception:
            logger.warning("cleanup after failed startup raised", exc_info=True)

    async def run(self, *, exit_watchdog: bool = False) -> None:
        """start(), wait for SIGINT/SIGTERM, then stop().

        The SIGINT handler goes in before start(): a plugin can raise SIGINT
        while it starts (the CLI read loop does when stdin is already at
        EOF), and under the default handler that interrupted startup mid-way
        or was lost. Installed first, it sets stop_event, startup finishes,
        and stop() runs. SIGTERM keeps its default action until startup is
        done, so `systemctl stop` during a hung startup still ends the
        process at once.

        With ``exit_watchdog`` true, the exit watchdog is armed once stop()
        returns or raises (see ``corvidae.shutdown``). Off by default so
        in-process callers (tests) are never force-exited.
        """
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        _shutdown_requested = False
        _starting = True

        def _handle_stop_signal() -> None:
            nonlocal _shutdown_requested
            if _shutdown_requested:
                sys.stderr.write("second interrupt received, force-exiting\n")
                sys.stderr.flush()
                os._exit(1)
            _shutdown_requested = True
            if _starting:
                # Say why nothing happens yet, and how to force it. stderr as
                # well as the log: `corvidae cli` logs to a file, and logging
                # may not be configured yet this early.
                message = (
                    "shutdown requested; waiting for startup to finish "
                    "(interrupt again to force)"
                )
                logger.info(message)
                sys.stderr.write(message + "\n")
                sys.stderr.flush()
            stop_event.set()

        loop.add_signal_handler(signal.SIGINT, _handle_stop_signal)

        try:
            await self.start()
        except BaseException:
            # Close whatever did start (persistence's aiosqlite connection has
            # a non-daemon thread that would otherwise hang interpreter exit),
            # then let the failure propagate so the process exits non-zero.
            # BaseException, not Exception: a CancelledError leaked from a
            # plugin's on_start skipped this cleanup and left the process
            # alive and silent under systemd (#282).
            await self._cleanup_failed_start()
            raise
        _starting = False
        loop.add_signal_handler(signal.SIGTERM, _handle_stop_signal)

        await stop_event.wait()

        logger.info("shutdown signal received, stopping")

        # Arm the watchdog even if stop() raises; the exception still propagates.
        try:
            await self.stop()
        finally:
            if exit_watchdog:
                arm_exit_watchdog(self.shutdown_timeouts.exit)


def run_daemon(runtime: Runtime) -> None:
    """Run ``runtime`` to completion with the exit watchdog armed.

    The single entry-point helper for serve, cli and acp, so none of them can
    hang on a leaked non-daemon thread after shutdown.
    """
    # Arming happens inside the coroutine so it also covers asyncio.run's
    # up-to-300 s default-executor join.
    asyncio.run(runtime.run(exit_watchdog=True))
