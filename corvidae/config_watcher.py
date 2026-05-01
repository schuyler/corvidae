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
    - Failed validation (no llm.main): logged at ERROR, reload skipped.
    - Per-plugin exceptions: logged at ERROR, other plugins still receive reload.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import yaml

from corvidae.channel import ChannelRegistry, load_channel_config
from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae.runtime import deep_merge

logger = logging.getLogger(__name__)

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
        3. Validate (must be a dict with llm.main). On failure: log ERROR, skip.
        4. Update ChannelRegistry.agent_defaults from the new agent config.
        5. Re-run load_channel_config for new channel entries.
        6. Dispatch on_config_reload to each plugin with per-plugin error isolation.
        """
        # Step 1: Parse YAML from disk.
        try:
            with open(self._config_path) as f:
                raw = yaml.safe_load(f)
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

        # Step 3: Validate — must have llm.main.
        llm_main = new_config.get("llm", {}).get("main")
        if not llm_main:
            logger.error(
                "config reload failed: llm.main is absent from %s after merge",
                self._config_path,
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

        # Step 5: Re-run load_channel_config to pick up new channel entries.
        # Existing channels are NOT updated (they retain their original ChannelConfig
        # until restart). load_channel_config is a no-op for channels that already exist.
        try:
            load_channel_config(new_config, registry)
        except ValueError:
            logger.error(
                "config reload: load_channel_config raised ValueError; "
                "channel config update skipped (agent defaults still applied)",
                exc_info=True,
            )

        # Step 6: Dispatch on_config_reload to each plugin individually.
        await self._dispatch_config_reload(new_config)

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
