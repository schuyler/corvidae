import logging
import time

import aiohttp

logger = logging.getLogger(__name__)


class LLMClient:
    """Async client for OpenAI-compatible Chat Completions API."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        """Create the aiohttp session."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self.session = aiohttp.ClientSession(headers=headers)

        logger.info(
            "client started",
            extra={"base_url": self.base_url, "model": self.model},
        )

    async def stop(self) -> None:
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()

        logger.info("client stopped")

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict:
        """Send a chat completion request.

        Returns the full response dict from the API.
        Raises aiohttp.ClientResponseError on non-2xx status.
        """
        if self.session is None:
            raise RuntimeError("LLMClient.start() must be called before chat()")

        logger.debug(
            "chat completion request",
            extra={
                "message_count": len(messages),
                "tool_count": len(tools) if tools else 0,
            },
        )

        payload = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools

        start = time.monotonic()
        async with self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
        ) as resp:
            try:
                resp.raise_for_status()
                response = await resp.json()
            except aiohttp.ClientResponseError as e:
                logger.error(
                    "chat completion failed",
                    extra={"status": resp.status},
                )
                raise
            elapsed = time.monotonic() - start

        usage = response.get("usage", {})
        logger.info(
            "chat completion returned",
            extra={
                "model": self.model,
                "latency_ms": round(elapsed * 1000, 1),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
        )

        return response
