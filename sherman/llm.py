import aiohttp


class LLMClient:
    """Async client for OpenAI-compatible Chat Completions API."""

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        """Create the aiohttp session."""
        self.session = aiohttp.ClientSession()

    async def stop(self) -> None:
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()

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
        payload = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
        async with self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
        ) as resp:
            resp.raise_for_status()
            return await resp.json()
