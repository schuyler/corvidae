"""LLMPlugin — owns LLM client lifecycle and configuration.

Parses llm.main and llm.background config, creates LLMClient instances,
manages their aiohttp session lifecycle. Other plugins access clients
via get_dependency(pm, "llm", LLMPlugin).

Config:
    llm:
      main:           # required
        base_url: ...
        model: ...
        api_key: ...        # optional
        extra_body: ...     # optional
        max_retries: 3      # optional
        retry_base_delay: 2.0
        retry_max_delay: 60.0
        timeout: ...        # optional
      background:     # optional — absent means use llm.main
        (same keys)
"""
import asyncio
import logging

from corvidae.hooks import CorvidaePlugin, hookimpl
from corvidae.llm import LLMClient

logger = logging.getLogger(__name__)


async def _close_after(client: LLMClient, delay: float = 5.0) -> None:
    """Close an old LLM client after a delay, allowing in-flight requests to finish."""
    await asyncio.sleep(delay)
    try:
        await client.stop()
    except Exception:
        logger.warning("failed to close old LLM client", exc_info=True)


class LLMPlugin(CorvidaePlugin):
    """Plugin that owns LLM client instances and their lifecycle."""

    depends_on = frozenset()

    def __init__(self, pm=None) -> None:
        if pm is not None:
            self.pm = pm
        self.main_client: LLMClient | None = None
        self.background_client: LLMClient | None = None
        self._main_config: dict | None = None
        self._bg_config: dict | None = None

    @hookimpl
    async def on_init(self, pm, config: dict) -> None:
        await super().on_init(pm, config)
        llm_config = config.get("llm", {})
        self._main_config = llm_config.get("main")
        self._bg_config = llm_config.get("background")

    @hookimpl
    async def on_start(self, config: dict) -> None:
        main_config = self._main_config
        if main_config is None:
            # Backward compat: read from config if on_init was not called
            main_config = config.get("llm", {}).get("main")
        if main_config is None:
            raise KeyError("llm.main config is required")
        self.main_client = self._create_client(main_config)
        await self.main_client.start()

        bg_config = self._bg_config
        if bg_config:
            self.background_client = self._create_client(bg_config)
            await self.background_client.start()

    @hookimpl
    async def on_stop(self) -> None:
        if self.main_client:
            await self.main_client.stop()
        if self.background_client:
            await self.background_client.stop()

    @hookimpl
    async def on_config_reload(self, config: dict) -> None:
        """Re-read llm config and swap main_client if it changed.

        Compares the new llm.main dict with the stored one. If changed,
        creates a new LLMClient, starts its session, and swaps the reference.
        The old client is closed asynchronously after a delay to allow
        in-flight requests to complete.
        """
        llm_config = config.get("llm", {})
        new_main_config = llm_config.get("main")
        if new_main_config is None:
            # Validation should have caught this; log and skip.
            logger.error("on_config_reload: llm.main missing from new config, skipping LLM swap")
            return

        # Compare new config with stored config; skip swap if unchanged.
        if new_main_config == self._main_config:
            return

        logger.info("on_config_reload: llm.main changed, creating new LLM client")

        # Create and start new client before swapping so that an error during
        # start does not leave the plugin with a None client.
        new_client = self._create_client(new_main_config)
        await new_client.start()

        # Swap the reference and store the updated config.
        old_client = self.main_client
        self.main_client = new_client
        self._main_config = new_main_config

        # Schedule deferred closure of the old client so in-flight requests finish.
        if old_client is not None:
            asyncio.create_task(_close_after(old_client, delay=5.0))

    def get_client(self, role: str = "main") -> LLMClient:
        """Return the client for the given role.

        Args:
            role: "main" or "background". Background falls back to main
                  if no background client is configured.
        """
        if role == "background":
            return self.background_client or self.main_client
        return self.main_client

    @staticmethod
    def _create_client(cfg: dict) -> LLMClient:
        return LLMClient(
            base_url=cfg["base_url"],
            model=cfg["model"],
            api_key=cfg.get("api_key"),
            extra_body=cfg.get("extra_body"),
            max_retries=cfg.get("max_retries", 3),
            retry_base_delay=cfg.get("retry_base_delay", 2.0),
            retry_max_delay=cfg.get("retry_max_delay", 60.0),
            timeout=cfg.get("timeout"),
        )
