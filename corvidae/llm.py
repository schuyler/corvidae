"""LLM client for OpenAI-compatible Chat Completions API.

This module provides an async HTTP client for LLM chat completions. It handles:
- Session lifecycle (start/stop with aiohttp)
- Request/response logging with timing and token usage
- Error handling and propagation

Logging:
    - INFO: client started/stopped, chat completion returned with metadata
    - DEBUG: chat completion request (message count, tool count)
    - ERROR: HTTP errors from the API

Token usage is extracted from the response's `usage` field. If the API doesn't
provide usage data, the fields are logged as `None`.
"""

import asyncio
import logging
import time

import aiohttp

TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}

logger = logging.getLogger(__name__)


class LLMClient:
    """Async client for OpenAI-compatible Chat Completions API.

    The client maintains an aiohttp session with optional Bearer auth.
    All requests include timing and token usage logging.

    Attributes:
        base_url: API base URL (e.g., "http://localhost:8080")
        model: Model name for requests (e.g., "llama3.1")
        api_key: Optional API key for Bearer auth
        session: aiohttp session (created by start(), closed by stop())
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        extra_body: dict | None = None,
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
        retry_max_delay: float = 60.0,
        timeout: float | None = None,
    ) -> None:
        """Initialize the LLM client.

        Args:
            base_url: API base URL (trailing slash will be stripped)
            model: Model identifier to use in requests
            api_key: Optional Bearer token for authentication
            extra_body: Optional dict of extra fields merged into every
                        chat request payload (instance-level defaults).
                        Call-level extra_body passed to chat() overrides
                        these when keys conflict.
            max_retries: Max retry attempts on transient errors (default 3).
                         Set to 0 to disable retries.
            retry_base_delay: Base delay in seconds for exponential backoff.
            retry_max_delay: Maximum delay cap in seconds.
            timeout: Total HTTP timeout in seconds per LLM request.
                     None uses aiohttp's session-level default (300s).
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.extra_body = extra_body
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self.timeout = (
            aiohttp.ClientTimeout(total=timeout) if timeout is not None else None
        )
        self.session: aiohttp.ClientSession | None = None

    def _retry_delay(
        self, attempt: int, retry_after: str | None = None
    ) -> float:
        """Compute retry delay with exponential backoff.

        Honors Retry-After header (integer/float seconds) when present.
        Falls back to exponential backoff for unparseable values.
        """
        if retry_after is not None:
            try:
                return min(float(retry_after), self.retry_max_delay)
            except ValueError:
                pass
        return min(self.retry_base_delay * (2 ** attempt), self.retry_max_delay)

    async def start(self) -> None:
        """Create the aiohttp session with optional auth header.

        Logs an INFO message with base_url and model.
        """
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self.session = aiohttp.ClientSession(headers=headers)

        logger.info(
            "client started",
            extra={"base_url": self.base_url, "model": self.model},
        )

    async def stop(self) -> None:
        """Close the aiohttp session.

        Logs an INFO message. Safe to call if session is None.
        """
        if self.session:
            await self.session.close()

        logger.info("client stopped")

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        extra_body: dict | None = None,
    ) -> dict:
        """Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            tools: Optional list of tool schemas for function calling
            extra_body: Optional dict of extra fields to merge into request payload

        Returns:
            Full response dict from the API (includes choices, usage, etc.)

        Raises:
            RuntimeError: If start() was not called
            aiohttp.ClientResponseError: On non-2xx HTTP status

        Logs:
            DEBUG: Request metadata (message count, tool count)
            INFO: Response metadata (latency_ms, token counts, model)
            ERROR: HTTP failure with status code
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
        if self.extra_body:
            payload.update(self.extra_body)  # instance defaults
        if extra_body:
            payload.update(extra_body)  # call-level override wins

        for attempt in range(1 + self.max_retries):
            try:
                start = time.monotonic()
                async with self.session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                ) as resp:
                    if (
                        resp.status in TRANSIENT_STATUS_CODES
                        and attempt < self.max_retries
                    ):
                        retry_after = resp.headers.get("Retry-After")
                        delay = self._retry_delay(attempt, retry_after)
                        logger.warning(
                            "transient LLM error, retrying",
                            extra={
                                "status": resp.status,
                                "attempt": attempt + 1,
                                "delay": delay,
                            },
                        )
                        await asyncio.sleep(delay)
                        continue

                    try:
                        resp.raise_for_status()
                        response = await resp.json()
                    except aiohttp.ClientResponseError:
                        logger.error(
                            "chat completion failed",
                            extra={"status": resp.status},
                        )
                        raise

            except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as exc:
                if attempt < self.max_retries:
                    delay = self._retry_delay(attempt)
                    logger.warning(
                        "transient LLM connection error, retrying",
                        extra={
                            "error": str(exc),
                            "attempt": attempt + 1,
                            "delay": delay,
                        },
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

            break  # success

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
