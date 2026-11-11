"""ConfigWatcherPlugin — hot-reload agent.yaml without restarting the daemon.

Polls the config file's mtime every `daemon.config_poll_interval` seconds
(default 2.0). On change, re-parses YAML, deep-merges CLI overrides, validates,
and dispatches `on_config_reload` to each plugin individually.

Config:
    daemon:
      config_poll_interval: 2.0   # optional; default 2.0 seconds

Reserved config keys consumed (set by Runtime):
    _config_path: pathlib.Path — path to agent.yaml
    _cli_overrides: dict — CLI overrides to re-apply on every reload

Error handling:
    - Invalid YAML: logged at ERROR, reload skipped.
    - Missing file: logged at WARNING, polling continues (file may reappear).
    - Failed validation (a malformed llm section, such as no default chat
      profile): logged at ERROR, reload skipped.
    - Per-plugin exceptions: logged at ERROR, other plugins still receive reload.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import yaml

from corvidae.channel import (
    ChannelRegistry,
    apply_channel_config,
    parse_channel_config,
    validate_agent_config,
)
from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae.idle import validate_daemon_config
from corvidae.llm_plugin import parse_llm_config
from corvidae.runtime import deep_merge
from corvidae.tool_collection import parse_read_file_window_fraction, parse_tool_timeouts

logger = logging.getLogger(__name__)

#: A key absent from a config. Distinct from a stored None, and distinct from
#: `_UNREPORTED` below: conflating the two silently suppressed the warning for
#: any key that was deleted, because an absent value compared equal to the
#: "never reported" default (#473).
_ABSENT = object()

#: No value has been reported for this key yet.
_UNREPORTED = object()


def _leaf_value(config: object, path: tuple[object, ...]) -> object:
    """The value at ``path`` (key names, outermost first), or ``_ABSENT``.

    The path is walked by key name rather than by re-splitting a dotted
    string: a key name may itself contain a dot (`llm.qwen3.6.model`), and
    splitting turned such a key into a path that never resolved, so its
    changes were never reported (#473).
    """
    current: object = config
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return _ABSENT
        current = current[part]
    return current


def _dotted(path: tuple[object, ...]) -> str:
    """A key path as the dotted name the warning names.

    Path tuples keep the keys as they were written, which may include a type
    YAML chose (`1`, `true`); the message needs a string, and sorting needs a
    key that compares across types.
    """
    return ".".join(str(part) for part in path)


def _changed_leaf_keys(
    old: dict, new: dict, prefix: tuple[object, ...] = ()
) -> set[tuple[object, ...]]:
    """Leaf key paths whose value differs between two config dicts.

    Paths are tuples of the keys themselves, so a dotted key name survives
    intact and a key YAML typed as a non-string still resolves. Stringifying
    here turned such a key into a name the lookup could never find, so every
    later comparison read `_ABSENT` and the change was suppressed (review). A
    whole subtree appearing or disappearing reports each leaf under it rather
    than the subtree name, so the warning names something the operator can act
    on.
    """
    changed: set[tuple[object, ...]] = set()
    for key in set(old) | set(new):
        path = (*prefix, key)
        in_old, in_new = key in old, key in new
        if in_old and in_new:
            a, b = old[key], new[key]
            if isinstance(a, dict) and isinstance(b, dict):
                changed |= _changed_leaf_keys(a, b, path)
            elif a != b:
                changed.add(path)
        else:
            value = new[key] if in_new else old[key]
            if isinstance(value, dict):
                changed |= _changed_leaf_keys(
                    {} if in_new else value, value if in_new else {}, path
                )
            else:
                changed.add(path)
    return changed


# Default poll interval in seconds if not specified in config.
_DEFAULT_POLL_INTERVAL = 2.0


class ConfigWatcherPlugin(CorvidaePlugin):
    """Plugin that watches agent.yaml for changes and reloads config on the fly."""

    depends_on = frozenset()

    def __init__(self) -> None:
        # _config_path and _cli_overrides are set in on_init from the config dict.
        self._config_path: Path | None = None
        self._cli_overrides: dict = {}
        self._poll_interval: float = _DEFAULT_POLL_INTERVAL

        # _last_mtime is set in on_start from the actual file stat.
        self._last_mtime: float = 0.0

        # _stop_event and _watcher_task are created in on_start.
        self._stop_event: asyncio.Event | None = None
        self._watcher_task: asyncio.Task | None = None

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        """Read config path, CLI overrides, and poll interval from config dict."""
        await super().on_init(pm, config)

        # Read reserved keys set by Runtime.start().
        self._config_path = config.get("_config_path")
        self._cli_overrides = config.get("_cli_overrides") or {}

        # Read poll interval from daemon section; fall back to default.
        daemon_config = config.get("daemon", {})
        self._poll_interval = daemon_config.get("config_poll_interval", _DEFAULT_POLL_INTERVAL)

    @hookimpl
    async def on_start(self, config: dict) -> None:
        """Initialize _last_mtime from actual file stat and launch the watch loop."""
        if self._config_path is None:
            logger.warning("No config_path configured; config watching disabled")
            return

        # Initialize _last_mtime to the current mtime to avoid a redundant reload
        # on the first poll cycle (the config was just loaded by Runtime.start).
        try:
            self._last_mtime = self._config_path.stat().st_mtime
        except FileNotFoundError:
            # File missing at startup; will be logged when polling begins.
            self._last_mtime = 0.0

        # Create stop event and launch the polling task.
        self._stop_event = asyncio.Event()
        self._watcher_task = asyncio.create_task(self._watch_loop())

    @hookimpl
    async def on_stop(self) -> None:
        """Cancel the watcher task and wait for it to finish."""
        if self._stop_event is not None:
            self._stop_event.set()

        if self._watcher_task is not None:
            self._watcher_task.cancel()
            try:
                await self._watcher_task
            except asyncio.CancelledError:
                pass
            self._watcher_task = None

    async def _watch_loop(self) -> None:
        """Poll the config file mtime and trigger reload when it changes.

        Runs until _stop_event is set or the task is cancelled. Uses
        `except Exception` intentionally — this does NOT catch
        asyncio.CancelledError (a BaseException in Python 3.8+), allowing
        task cancellation from on_stop to propagate correctly. Do not
        "fix" this by catching BaseException.
        """
        while not self._stop_event.is_set():
            try:
                current_mtime = self._config_path.stat().st_mtime
                if current_mtime != self._last_mtime:
                    # Mtime changed; reload and update the stored mtime.
                    await self._reload_config()
                    self._last_mtime = current_mtime
            except FileNotFoundError:
                # Config file was deleted; log and wait for it to reappear.
                logger.warning("config file not found: %s", self._config_path)
            except Exception:
                # NOTE: `except Exception` intentionally does NOT catch
                # asyncio.CancelledError (a BaseException in Python 3.8+).
                # This allows task cancellation from on_stop to propagate
                # correctly. Do not "fix" this by catching BaseException.
                logger.exception("config watcher error")
            await asyncio.sleep(self._poll_interval)

    async def _reload_config(self) -> None:
        """Read, validate, and apply the updated config file.

        Steps:
        1. Parse YAML from disk. On failure: log ERROR, skip.
        2. Deep-merge CLI overrides on top of the new YAML.
        3. Validate the llm section (parse_llm_config). On failure: log ERROR, skip.
        3b-3e. Validate agent, daemon, channels and tools (call_timeout,
           timeouts) sections. On failure: log ERROR, skip the whole reload
           (nothing is applied, #208).
        4. Update ChannelRegistry.agent_defaults from the new agent config.
        5. Apply the parsed channel entries (cannot fail validation now).
        6. Dispatch on_config_reload to each plugin with per-plugin error isolation.
        """
        # Step 1: Parse YAML from disk.
        # Decoded explicitly, as at startup: UTF-8 whatever the locale, and a
        # decode error's offset counts from the start of the file.
        try:
            text = Path(self._config_path).read_bytes().decode("utf-8")
            raw = yaml.safe_load(text)
        except UnicodeDecodeError as exc:
            # Name the cause and the fix rather than "could not read" (#310).
            logger.error(
                "config reload failed: %s is not valid UTF-8 (%s at byte %d); "
                "re-save it as UTF-8; the running config stays in force",
                self._config_path, exc.reason, exc.start, exc_info=True,
            )
            return
        except RecursionError:
            logger.error(
                "config reload failed: YAML in %s is nested too deeply to load; "
                "reduce the nesting; the running config stays in force",
                self._config_path, exc_info=True,
            )
            return
        except yaml.YAMLError:
            logger.exception("config reload failed: invalid YAML in %s", self._config_path)
            return
        except Exception:
            logger.exception("config reload failed: could not read %s", self._config_path)
            return

        # Step 2: Deep-merge CLI overrides on top of new YAML.
        if not isinstance(raw, dict):
            logger.error(
                "config reload failed: YAML did not produce a dict (got %s)",
                type(raw).__name__,
            )
            return

        new_config = deep_merge(raw, self._cli_overrides)

        # Preserve the reserved keys that Runtime sets at startup.
        new_config["_config_path"] = self._config_path
        new_config["_cli_overrides"] = self._cli_overrides
        if hasattr(self, "config") and self.config is not None:
            base_dir = self.config.get("_base_dir")
            if base_dir is not None:
                new_config["_base_dir"] = base_dir

        # Step 3: Validate the llm section. Nothing in it is applied on
        # reload, but a malformed one refuses the whole reload.
        try:
            parse_llm_config(new_config.get("llm"))
        except ValueError as exc:
            logger.error(
                "config reload failed: %s in %s; reload skipped, running config unchanged",
                exc, self._config_path, exc_info=True,
            )
            return

        # Step 3b: Validate the agent section before it becomes agent_defaults —
        # an invalid reasoning_history would otherwise make every later
        # resolve_config raise.
        try:
            validate_agent_config(new_config.get("agent", {}))
        except ValueError:
            logger.exception("config reload failed: invalid agent config in %s", self._config_path)
            return

        # Step 3c: Reject removed daemon idle keys (#165), same as startup.
        try:
            validate_daemon_config(new_config.get("daemon") or {})
        except ValueError:
            logger.exception("config reload failed: invalid daemon config in %s", self._config_path)
            return

        # Step 3d: Validate the channels section without applying it, so a
        # rejected entry leaves agent_defaults and everything else untouched
        # ("a malformed reload changes nothing", #208).
        try:
            parsed_channels = parse_channel_config(new_config)
        except ValueError as exc:
            logger.error(
                "config reload failed: %s in %s; reload skipped, running config unchanged",
                exc, self._config_path, exc_info=True,
            )
            return

        # Step 3e: Validate tools.call_timeout, tools.timeouts and
        # tools.read_file_window_fraction. The keys stay restart-only; a bad
        # value refuses the whole reload.
        try:
            parse_tool_timeouts(new_config.get("tools"))
            parse_read_file_window_fraction(new_config.get("tools"))
        except ValueError as exc:
            logger.error(
                "config reload failed: %s in %s; reload skipped, running config unchanged",
                exc, self._config_path, exc_info=True,
            )
            return

        # Step 4: Update agent_defaults on both the class and the registered instance.
        # The class-level attribute is updated so that test code and any references
        # to ChannelRegistry.agent_defaults (class-level) see the new value.
        # The instance attribute is updated so the runtime registry reflects the change
        # on resolve_config calls.
        new_agent_defaults = new_config.get("agent", {})
        ChannelRegistry.agent_defaults = new_agent_defaults
        registry = self.pm.get_plugin("registry")
        if registry is not None:
            registry.agent_defaults = new_agent_defaults

        # Step 5: Apply the already-validated channel entries. Existing
        # channels keep their own ChannelConfig until restart, except
        # tools.disabled; transport-wide entries ("signal:*") are replaced and
        # re-matched against every live channel, like agent defaults above.
        apply_channel_config(parsed_channels, registry)

        # Step 5b: Report changes no plugin reloads (they wait for a restart).
        self._warn_restart_only_changes(new_config)

        # Step 6: Dispatch on_config_reload to each plugin individually.
        await self._dispatch_config_reload(new_config)

    def _reloadable_prefixes(self) -> tuple[str, ...]:
        """Every key prefix some loaded plugin declares it re-reads.

        Plugins opt in with `reloadable_keys`; anything not claimed is
        restart-only, which is the safe default (#408).
        """
        prefixes: list[str] = []
        for plugin in self.pm.get_plugins():
            prefixes.extend(getattr(plugin, "reloadable_keys", ()) or ())
        return tuple(prefixes)

    def _warn_restart_only_changes(self, new_config: dict) -> None:
        """Log one WARNING per changed key that no plugin reloads.

        Without this a restart-only edit is silent: the reload reports the
        plugins it did update, and the operator learns the change was pending
        only by noticing behaviour never changed (#408).

        The comparison is against the config the running daemon is using
        (`self.config`, set once in on_init and never advanced), because the
        warning claims what a restart would change. Diffing against the
        previous reload reported a key edited back to its running value as
        pending, when the file and the process already agreed (#473).

        A key whose pending value is unchanged from the last report is not
        reported again, so a parked file does not warn on every poll. The
        record is rebuilt each pass, so a key that reverts is forgotten and a
        later edit to it is news again.
        """
        # Each declared prefix as its own key segments. A changed key is
        # matched on its path, never on a re-joined dotted string: a key whose
        # *name* contains a dot ("agent.forks") joins to the same string as a
        # nested path and was mistaken for it (#591).
        live = [tuple(prefix.split(".")) for prefix in self._reloadable_prefixes()]
        running = self.config if isinstance(getattr(self, "config", None), dict) else {}
        reported = getattr(self, "_reported_pending", None)
        if not isinstance(reported, dict):
            reported = {}
        still_pending: dict[tuple[object, ...], object] = {}

        for path in sorted(_changed_leaf_keys(running, new_config), key=_dotted):
            dotted = _dotted(path)
            if dotted.startswith("_") or any(
                tuple(map(str, path[: len(prefix)])) == prefix for prefix in live
            ):
                continue
            # `_ABSENT` here means the key was deleted; `_UNREPORTED` means
            # nothing was reported for it yet. They must differ, or a deletion
            # compares equal to the default and is silently never reported.
            value = _leaf_value(new_config, path)
            if reported.get(path, _UNREPORTED) != value:
                logger.warning(
                    "config reload: %s changed; restart-only, takes effect "
                    "on restart",
                    dotted,
                    extra={"key": dotted},
                )
            still_pending[path] = value

        self._reported_pending = still_pending

    async def _dispatch_config_reload(self, config: dict) -> None:
        """Dispatch on_config_reload to each plugin with per-plugin error isolation.

        Bypasses pm.ahook to avoid asyncio.gather, which would abort all
        remaining plugins if one raises (apluggy broadcast uses gather without
        return_exceptions=True). Iterates hookimpls individually so that a
        failing plugin logs an error but does not block the others.

        Filters kwargs by hook_impl.argnames, matching pluggy's _multicall
        behavior: hookimpls that omit 'config' from their signature receive
        no arguments rather than raising TypeError.
        """
        hook = self.pm.hook.on_config_reload
        hookimpls = hook.get_hookimpls()
        caller_kwargs = {"config": config}

        # Iterate in reverse (pluggy calls hookimpls in LIFO order by default).
        for hook_impl in reversed(hookimpls):
            # Filter caller_kwargs to only the args the hookimpl declared.
            impl_kwargs = {
                k: caller_kwargs[k]
                for k in hook_impl.argnames
                if k in caller_kwargs
            }
            try:
                await hook_impl.function(**impl_kwargs)
            except Exception:
                logger.exception(
                    "on_config_reload failed for plugin %s",
                    hook_impl.plugin_name,
                )
