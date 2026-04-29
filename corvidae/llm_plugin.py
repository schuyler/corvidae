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
import logging

from corvidae.hooks import hookimpl
from corvidae.llm import LLMClient

logger = logging.getLogger(__name__)


class LLMPlugin:
    """Plugin that owns LLM client instances and their lifecycle."""

    depends_on = set()

    def __init__(self, pm) -> None:
        self.pm = pm
        self.main_client: LLMClient | None = None
        self.background_client: LLMClient | None = None

    @hookimpl
    async def on_start(self, config: dict) -> None:
        llm_config = config.get("llm", {})
        main_config = llm_config["main"]  # required
        self.main_client = self._create_client(main_config)
        await self.main_client.start()

        bg_config = llm_config.get("background")
        if bg_config:
            self.background_client = self._create_client(bg_config)
            await self.background_client.start()

    @hookimpl
    async def on_stop(self) -> None:
        if self.main_client:
            await self.main_client.stop()
        if self.background_client:
            await self.background_client.stop()

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
